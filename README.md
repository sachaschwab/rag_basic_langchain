# Legal Document QA System

This project is a Legal Document Question-Answering system that uses natural language processing and machine learning to answer questions based on a corpus of legal documents.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features
- Document processing for PDF, HTML, and DOCX files
- Text splitting for efficient processing
- Vector store creation using OpenAI embeddings
- Conversational retrieval chain for question answering
- Persistent storage of vector embeddings
- References provided with answers for traceability

## Installation
1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install langchain chromadb openai tiktoken unstructured
   ```
3. Set up your OpenAI API key as an environment variable or be prepared to enter it when prompted.

## Usage
1. Place your legal documents in a 'documents' directory in the project root.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Choose whether to load documents newly or use an existing vector store.
4. Enter your legal question when prompted.

Example usage:

## Contributing
Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License
This project is licensed under the MIT License.
