const fileInput = document.getElementById('json-input');
const processButton = document.getElementById('process-button');
const outputDisplay = document.getElementById('output-data');

async function loadJsonFile() {
  return new Promise((resolve, reject) => {
    const file = document.getElementById('json-input').files[0];
    
    if (!file) {
      return reject(new Error("No file selected."));
    }
    
    const reader = new FileReader();

    reader.onload = function(e) {
      try {
        const data = JSON.parse(e.target.result);
        resolve(data); // Resolve the promise with the parsed object
      } catch (parseError) {
        reject(new Error("Invalid JSON format.")); // Reject if parsing fails
      }
    };
    
    reader.onerror = function() {
        reject(new Error("Error reading the file.")); // Reject if read fails
    };

    reader.readAsText(file);
  });
}
