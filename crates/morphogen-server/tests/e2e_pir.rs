use morphogen_client::{aggregate_responses, generate_query, EpochMetadata, ServerResponse};
use morphogen_core::{CuckooTable, DeltaBuffer};
use morphogen_dpf::AesDpfKey;
use morphogen_storage::ChunkedMatrix;

const ROW_SIZE: usize = 1024;
const NUM_ROWS: usize = 1000;
const CHUNK_SIZE: usize = 64 * 1024;

fn create_matrix_with_payload(target_row: usize, payload: &[u8]) -> ChunkedMatrix {
    let total_size = NUM_ROWS * ROW_SIZE;
    let mut matrix = ChunkedMatrix::new(total_size, CHUNK_SIZE);
    matrix.fill_with_pattern(0xDEADBEEF);
    matrix.write_row(target_row, ROW_SIZE, payload);
    matrix
}

#[test]
fn e2e_two_server_pir_recovers_payload() {
    let target_row = 42;
    let payload = vec![0xCA; ROW_SIZE];

    let matrix_a = create_matrix_with_payload(target_row, &payload);
    let matrix_b = create_matrix_with_payload(target_row, &payload);
    let delta_a = DeltaBuffer::new(ROW_SIZE);
    let delta_b = DeltaBuffer::new(ROW_SIZE);

    let mut rng = rand::thread_rng();
    let (key0_a, key0_b) = AesDpfKey::generate_pair(&mut rng, target_row);
    let (key1_a, key1_b) = AesDpfKey::generate_pair(&mut rng, NUM_ROWS + 100);
    let (key2_a, key2_b) = AesDpfKey::generate_pair(&mut rng, NUM_ROWS + 200);

    let keys_a = [key0_a, key1_a, key2_a];
    let keys_b = [key0_b, key1_b, key2_b];

    let results_a = morphogen_server::try_scan(&matrix_a, &delta_a, &keys_a, ROW_SIZE).unwrap();
    let results_b = morphogen_server::try_scan(&matrix_b, &delta_b, &keys_b, ROW_SIZE).unwrap();

    let mut recovered = vec![0u8; ROW_SIZE];
    for i in 0..ROW_SIZE {
        recovered[i] = results_a[0][i] ^ results_b[0][i];
    }

    assert_eq!(
        recovered, payload,
        "recovered payload should match original"
    );
}

#[test]
fn e2e_pir_with_delta_buffer() {
    let target_row = 42;
    let original_payload = vec![0xAA; ROW_SIZE];
    let delta_diff = vec![0x55; ROW_SIZE];

    let matrix_a = create_matrix_with_payload(target_row, &original_payload);
    let matrix_b = create_matrix_with_payload(target_row, &original_payload);

    let delta_a = DeltaBuffer::new(ROW_SIZE);
    let delta_b = DeltaBuffer::new(ROW_SIZE);
    // Push update to both servers
    delta_a.push(target_row, delta_diff.clone()).unwrap();
    delta_b.push(target_row, delta_diff.clone()).unwrap();

    let mut rng = rand::thread_rng();
    let (key0_a, key0_b) = AesDpfKey::generate_pair(&mut rng, target_row);
    let (key1_a, key1_b) = AesDpfKey::generate_pair(&mut rng, NUM_ROWS + 100);
    let (key2_a, key2_b) = AesDpfKey::generate_pair(&mut rng, NUM_ROWS + 200);

    let keys_a = [key0_a, key1_a, key2_a];
    let keys_b = [key0_b, key1_b, key2_b];

    let results_a = morphogen_server::try_scan(&matrix_a, &delta_a, &keys_a, ROW_SIZE).unwrap();
    let results_b = morphogen_server::try_scan(&matrix_b, &delta_b, &keys_b, ROW_SIZE).unwrap();

    let mut recovered = vec![0u8; ROW_SIZE];
    for i in 0..ROW_SIZE {
        recovered[i] = results_a[0][i] ^ results_b[0][i];
    }

    let mut expected = vec![0u8; ROW_SIZE];
    for i in 0..ROW_SIZE {
        expected[i] = original_payload[i] ^ delta_diff[i];
    }

    assert_eq!(
        recovered, expected,
        "recovered payload should include delta XOR"
    );
}

#[test]
fn e2e_three_parallel_queries_recover_different_rows() {
    let targets = [10, 500, 999];
    let payloads: [Vec<u8>; 3] = [
        vec![0x11; ROW_SIZE],
        vec![0x22; ROW_SIZE],
        vec![0x33; ROW_SIZE],
    ];

    let total_size = NUM_ROWS * ROW_SIZE;
    let mut matrix_a = ChunkedMatrix::new(total_size, CHUNK_SIZE);
    let mut matrix_b = ChunkedMatrix::new(total_size, CHUNK_SIZE);
    matrix_a.fill_with_pattern(0x12345678);
    matrix_b.fill_with_pattern(0x12345678);

    for (i, &target) in targets.iter().enumerate() {
        matrix_a.write_row(target, ROW_SIZE, &payloads[i]);
        matrix_b.write_row(target, ROW_SIZE, &payloads[i]);
    }

    let delta_a = DeltaBuffer::new(ROW_SIZE);
    let delta_b = DeltaBuffer::new(ROW_SIZE);

    let mut rng = rand::thread_rng();
    let (key0_a, key0_b) = AesDpfKey::generate_pair(&mut rng, targets[0]);
    let (key1_a, key1_b) = AesDpfKey::generate_pair(&mut rng, targets[1]);
    let (key2_a, key2_b) = AesDpfKey::generate_pair(&mut rng, targets[2]);

    let keys_a = [key0_a, key1_a, key2_a];
    let keys_b = [key0_b, key1_b, key2_b];

    let results_a = morphogen_server::try_scan(&matrix_a, &delta_a, &keys_a, ROW_SIZE).unwrap();
    let results_b = morphogen_server::try_scan(&matrix_b, &delta_b, &keys_b, ROW_SIZE).unwrap();

    for i in 0..3 {
        let mut recovered = vec![0u8; ROW_SIZE];
        for j in 0..ROW_SIZE {
            recovered[j] = results_a[i][j] ^ results_b[i][j];
        }
        assert_eq!(
            recovered, payloads[i],
            "query {} should recover payload for row {}",
            i, targets[i]
        );
    }
}

#[test]
fn e2e_cuckoo_addressed_pir() {
    let num_accounts = 100;
    let table_size = num_accounts * 2;

    let mut cuckoo: CuckooTable<Vec<u8>> = CuckooTable::new(table_size);
    let addresser = cuckoo.addresser();

    let mut account_payloads: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for i in 0..num_accounts {
        let account_key = format!("0x{:040x}", i).into_bytes();
        let payload = vec![(i & 0xFF) as u8; ROW_SIZE];
        cuckoo.insert(account_key.clone(), payload.clone()).unwrap();
        account_payloads.push((account_key, payload));
    }

    let total_size = table_size * ROW_SIZE;
    let mut matrix_a = ChunkedMatrix::new(total_size, CHUNK_SIZE);
    let mut matrix_b = ChunkedMatrix::new(total_size, CHUNK_SIZE);
    matrix_a.fill_with_pattern(0xABCDEF);
    matrix_b.fill_with_pattern(0xABCDEF);

    for i in 0..num_accounts {
        let key = format!("0x{:040x}", i).into_bytes();
        if let Some((idx, payload)) = cuckoo.get(&key) {
            if idx != usize::MAX {
                matrix_a.write_row(idx, ROW_SIZE, payload);
                matrix_b.write_row(idx, ROW_SIZE, payload);
            }
        }
    }

    let delta_a = DeltaBuffer::new(ROW_SIZE);
    let delta_b = DeltaBuffer::new(ROW_SIZE);

    let target_account = b"0x0000000000000000000000000000000000000042".to_vec();
    let expected_payload = cuckoo.get(&target_account).unwrap().1.clone();
    let candidate_indices = addresser.hash_indices(&target_account);

    let mut rng = rand::thread_rng();
    let (key0_a, key0_b) = AesDpfKey::generate_pair(&mut rng, candidate_indices[0]);
    let (key1_a, key1_b) = AesDpfKey::generate_pair(&mut rng, candidate_indices[1]);
    let (key2_a, key2_b) = AesDpfKey::generate_pair(&mut rng, candidate_indices[2]);

    let keys_a = [key0_a, key1_a, key2_a];
    let keys_b = [key0_b, key1_b, key2_b];

    let results_a = morphogen_server::try_scan(&matrix_a, &delta_a, &keys_a, ROW_SIZE).unwrap();
    let results_b = morphogen_server::try_scan(&matrix_b, &delta_b, &keys_b, ROW_SIZE).unwrap();

    let mut found_payload = false;
    for i in 0..3 {
        let mut recovered = vec![0u8; ROW_SIZE];
        for j in 0..ROW_SIZE {
            recovered[j] = results_a[i][j] ^ results_b[i][j];
        }
        if recovered == expected_payload {
            found_payload = true;
            break;
        }
    }

    assert!(
        found_payload,
        "should recover payload from one of the 3 Cuckoo candidate positions"
    );
}

#[test]
fn e2e_full_client_server_flow() {
    let num_accounts = 100;
    let table_size = num_accounts * 2;
    let seeds: [u64; 3] = [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98];

    let mut cuckoo: CuckooTable<Vec<u8>> = CuckooTable::with_seeds(table_size, seeds);

    for i in 0..num_accounts {
        let account_key = format!("account_{}", i).into_bytes();
        let payload = vec![(i & 0xFF) as u8; ROW_SIZE];
        cuckoo.insert(account_key.clone(), payload).unwrap();
    }

    let total_size = table_size * ROW_SIZE;
    let mut matrix_a = ChunkedMatrix::new(total_size, CHUNK_SIZE);
    let mut matrix_b = ChunkedMatrix::new(total_size, CHUNK_SIZE);
    matrix_a.fill_with_pattern(0xCAFEBABE);
    matrix_b.fill_with_pattern(0xCAFEBABE);

    for i in 0..num_accounts {
        let key = format!("account_{}", i).into_bytes();
        if let Some((idx, payload)) = cuckoo.get(&key) {
            if idx != usize::MAX {
                matrix_a.write_row(idx, ROW_SIZE, payload);
                matrix_b.write_row(idx, ROW_SIZE, payload);
            }
        }
    }

    let delta_a = DeltaBuffer::new(ROW_SIZE);
    let delta_b = DeltaBuffer::new(ROW_SIZE);

    let metadata = EpochMetadata {
        epoch_id: 1,
        num_rows: table_size,
        seeds,
        block_number: 12345,
        state_root: [0u8; 32],
    };

    let target_account = b"account_42";
    let expected_payload = cuckoo.get(target_account).unwrap().1.clone();

    let mut rng = rand::thread_rng();
    let query = generate_query(&mut rng, target_account, &metadata);

    let results_a =
        morphogen_server::try_scan(&matrix_a, &delta_a, &query.keys_a, ROW_SIZE).unwrap();
    let results_b =
        morphogen_server::try_scan(&matrix_b, &delta_b, &query.keys_b, ROW_SIZE).unwrap();

    let response_a = ServerResponse {
        epoch_id: metadata.epoch_id,
        payloads: [
            results_a[0].clone(),
            results_a[1].clone(),
            results_a[2].clone(),
        ],
    };
    let response_b = ServerResponse {
        epoch_id: metadata.epoch_id,
        payloads: [
            results_b[0].clone(),
            results_b[1].clone(),
            results_b[2].clone(),
        ],
    };

    let aggregated = aggregate_responses(&response_a, &response_b).unwrap();
    assert_eq!(aggregated.epoch_id, metadata.epoch_id);

    let mut found = false;
    for payload in &aggregated.payloads {
        if payload == &expected_payload {
            found = true;
            break;
        }
    }

    assert!(
        found,
        "client should recover the correct payload using generate_query + aggregate_responses"
    );
}
