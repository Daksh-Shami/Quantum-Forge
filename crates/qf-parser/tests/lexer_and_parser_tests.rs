use glob::glob;
use oq3_parser::Step;
use qf_parser::parse_qasm_file;

#[test]
fn test_all_qasm_files() {
    let pattern = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/QASM_files/*.qasm"
    );

    // Find all .qasm files in the directory
    let paths = glob(pattern).expect("Failed to read glob pattern");
    let mut tested_files = 0;

    for path in paths {
        match path {
            Ok(path) => {
                let file_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");

                println!("Testing file: {}", file_name);

                // Parse the file and get the Output
                let parsed = parse_qasm_file(path.to_str().unwrap()).unwrap();

                // Print only parser steps that include errors
                for step in parsed.iter() {
                    match step {
                        Step::Error { msg } => println!("  Error: {}", msg),
                        _ => (),
                    }
                }

                // Collect any error messages from the parser output
                let errors: Vec<_> = parsed
                    .iter()
                    .filter_map(|step| {
                        if let Step::Error { msg } = step {
                            Some(msg)
                        } else {
                            None
                        }
                    })
                    .collect();

                assert!(
                    errors.is_empty(),
                    "Parser produced errors for {}: {:?}",
                    file_name,
                    errors
                );
                tested_files += 1;
            }
            Err(e) => println!("Error processing path: {}", e),
        }
    }

    assert!(tested_files > 0, "No QASM files were found to test!");
}
