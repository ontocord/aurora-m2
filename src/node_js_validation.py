import subprocess
import json
import tempfile
import os

def check_js_syntax(js_code):
    """
    Check JavaScript code for syntax errors using Node.js
    
    Args:
        js_code (str): JavaScript code to check
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    # Create a temporary file to store the JavaScript code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
        temp_file.write(js_code)
        temp_filename = temp_file.name

    try:
        # Use Node.js to parse the JavaScript code
        process = subprocess.run(
            ['node', '--check', temp_filename],
            capture_output=True,
            text=True
        )
        
        # Check if there were any syntax errors
        if process.returncode == 0:
            return True, "JavaScript code is syntactically valid"
        else:
            return False, process.stderr.strip()
            
    except subprocess.CalledProcessError as e:
        return False, f"Error running Node.js: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)

# Example usage
def test_js_syntax():
    # Test case 1: Valid JavaScript
    valid_js = """
    function add(a, b) {
        return a + b;
    }
    const result = add(5, 3);
    console.log(result);
    """
    
    # Test case 2: Invalid JavaScript (missing parenthesis)
    invalid_js = """
    function add(a, b {
        return a + b;
    }
    """
    
    # Run tests
    print("Testing valid JavaScript:")
    is_valid, message = check_js_syntax(valid_js)
    print(f"Valid: {is_valid}")
    print(f"Message: {message}\n")
    
    print("Testing invalid JavaScript:")
    is_valid, message = check_js_syntax(invalid_js)
    print(f"Valid: {is_valid}")
    print(f"Message: {message}")

if __name__ == "__main__":
    test_js_syntax()
