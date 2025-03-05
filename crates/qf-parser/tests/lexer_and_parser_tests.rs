use qf_parser::parse_qasm_file;
use oq3_parser::Step;

#[test]
fn test_lexer_parser_pipeline() {
    let file_path = "../../examples/QASM_files/simple_circuit.qasm";
    
    // Parse the file and get the Output.
    let parsed = parse_qasm_file(file_path).unwrap();
    
    // Print only parser steps that include errors.
    for step in parsed.iter() {
        match step {
            Step::Error { msg } => println!("  Error: {}", msg),
            _ => (),
        }
    }
    
    // Collect any error messages from the parser output.
    let errors: Vec<_> = parsed.iter()
        .filter_map(|step| {
            if let Step::Error { msg } = step {
                Some(msg)
            } else {
                None
            }
        })
        .collect();
    
    assert!(errors.is_empty(), "Parser produced errors: {:?}", errors);
}
