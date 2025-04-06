# QF Frontend

A Tauri-based GUI frontend for the QF Compiler project. This crate integrates with the rest of the QF Compiler workspace to provide a desktop application interface.

## Directory Structure

- **src/** - Contains the Rust source code for the Tauri application
  - `main.rs` - Entry point for the Tauri application
- **dist/** - Contains the frontend web assets (HTML, CSS, JS)
  - `index.html` - Main HTML file loaded by Tauri
- **capabilities/** - Contains capability definitions for Tauri v2
  - `default.json` - Defines the default capabilities for the application
- **build.rs** - Build script for the Tauri application
- **tauri.conf.json** - Configuration file for Tauri
- **icons/** - Contains icon files for the application

## Setup

This crate is part of the QF Compiler workspace. The frontend uses Tauri v2 to create a desktop application with a web-based UI.

### Prerequisites

- Rust toolchain
- Tauri CLI: `cargo install tauri-cli`

### Development

To run the development server:

```bash
cargo tauri dev -c crates/qf-frontend/tauri.conf.json
```

This will compile the Rust code and start the Tauri application with hot reloading enabled.

### Building

To build the application for production:

```bash
cargo tauri build -c crates/qf-frontend/tauri.conf.json
```

This will create a distributable package in the `target/release` directory.

## Configuration

### Tauri Configuration

The `tauri.conf.json` file contains the configuration for the Tauri application:

- **productName**: The name of the application
- **version**: The version of the application
- **identifier**: The unique identifier for the application
- **frontendDist**: The location of the frontend assets
- **windows**: Configuration for the application windows
- **security**: Security settings for the application

### Capabilities

Tauri v2 uses a capability-based security model. The `capabilities/default.json` file defines the capabilities of the application:

- **shell:allow:open**: Allows the application to open URLs in the default browser

## Frontend Development

The frontend is a simple web application located in the `dist` directory. To develop the frontend:

1. Edit the files in the `dist` directory
2. Run the Tauri development server to see your changes

## Integration with QF Compiler

This frontend is designed to integrate with the QF Compiler project. Future development will include:

- Visualization of quantum circuits
- Interactive quantum programming interface
- Integration with the QF Compiler backend
- Result visualization and analysis tools
