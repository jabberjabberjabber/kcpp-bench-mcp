import argparse
import json
import struct
import os
from pathlib import Path


def read_4_byte(file_object) -> bytes:
    """Read 4 bytes from file"""
    assert file_object.mode == "rb"
    return file_object.read(4)


def read_8_byte(file_object) -> bytes:
    """Read 8 bytes from file"""
    assert file_object.mode == "rb"
    return file_object.read(8)


def struct_unpack_uint32(data):
    """Unpack uint32"""
    assert len(data) == 4
    return struct.unpack("I", data)[0]


def struct_unpack_uint64(data):
    """Unpack uint64"""
    assert len(data) == 8
    return struct.unpack("Q", data)[0]


class GGUFString:
    """GGUF string reader"""
    
    @classmethod
    def read(cls, f):
        """Read GGUF string from file"""
        # Read string length (uint64)
        length_data = read_8_byte(f)
        length = struct_unpack_uint64(length_data)
        
        # Read string content
        data = f.read(length)
        return data.decode("utf-8")


def extract_minimal_info(filename: str) -> dict:
    """Extract only essential model information for optimization"""
    
    result = {
        "path": str(filename),
        "total_layers": 0,
        "model_name": "unknown",
        "context_length": 0,
        "architecture": "unknown",
    }
    
    try:
        with open(filename, "rb") as f:
            # Read GGUF header
            header_data = f.read(4 + 4 + 8 + 8)
            
            if len(header_data) < 24:
                result["error"] = "File too small to be valid GGUF"
                return result
            
            # Unpack header: magic, version, tensor_count, metadata_kv_count
            magic, version, tensor_count, metadata_kv_count = struct.unpack("IIQQ", header_data)
            
            # Verify magic number
            if magic != 0x46554747:  # "GGUF" in hex
                result["error"] = f"Invalid GGUF magic number: {hex(magic)}"
                return result
            
            # Read all metadata in one pass
            for _ in range(metadata_kv_count):
                # Read key
                key = GGUFString.read(f)
                
                # Read value type (uint32)
                value_type_data = read_4_byte(f)
                value_type = struct_unpack_uint32(value_type_data)
                
                # Extract all values we need
                if key == "general.architecture":
                    if value_type == 8:  # STRING
                        result["architecture"] = GGUFString.read(f)
                    else:
                        _skip_value(f, value_type)
                        
                        
                elif key in ["general.name"]:
                    if value_type == 8:  # STRING
                        result["model_name"] = GGUFString.read(f)
                    else:
                        _skip_value(f, value_type)
                        
                # Architecture-specific keys (we'll check these dynamically)
                elif ".block_count" in key:
                    if value_type == 4:  # UINT32
                        layer_data = read_4_byte(f)
                        result["total_layers"] = struct_unpack_uint32(layer_data)
                    else:
                        _skip_value(f, value_type)
                        
                elif ".context_length" in key:
                    if value_type == 4:  # UINT32
                        ctx_data = read_4_byte(f)
                        result["context_length"] = struct_unpack_uint32(ctx_data)
                    else:
                        _skip_value(f, value_type)
                        
                else:
                    # Skip other metadata
                    _skip_value(f, value_type)
        
        # Validation - only set error if there are actual problems
        if result["total_layers"] == 0:
            result["error"] = f"Could not find block_count for architecture '{result['architecture']}'"
            
        if result["context_length"] == 0:
            result["error"] = f"Could not find context_length for architecture '{result['architecture']}'"
                
    except Exception as e:
        result["error"] = str(e)
    
    return result


def _skip_value(f, value_type):
    """Skip a value based on its type"""
    if value_type == 0:  # UINT8
        f.read(1)
    elif value_type == 1:  # INT8
        f.read(1)
    elif value_type == 2:  # UINT16
        f.read(2)
    elif value_type == 3:  # INT16
        f.read(2)
    elif value_type == 4:  # UINT32
        f.read(4)
    elif value_type == 5:  # INT32
        f.read(4)
    elif value_type == 6:  # FLOAT32
        f.read(4)
    elif value_type == 7:  # BOOL
        f.read(1)
    elif value_type == 8:  # STRING
        GGUFString.read(f)
    elif value_type == 9:  # ARRAY
        _skip_array(f)
    elif value_type == 10:  # UINT64
        f.read(8)
    elif value_type == 11:  # INT64
        f.read(8)
    elif value_type == 12:  # FLOAT64
        f.read(8)
    else:
        raise ValueError(f"Unknown value type: {value_type}")


def _skip_array(f):
    """Skip an array value"""
    # Read array element type
    element_type_data = read_4_byte(f)
    element_type = struct_unpack_uint32(element_type_data)
    
    # Read array length
    length_data = read_8_byte(f)
    length = struct_unpack_uint64(length_data)
    
    # Skip all elements
    for _ in range(length):
        _skip_value(f, element_type)


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Extract minimal GGUF model info for optimization"
    )
    parser.add_argument("filename", help="Path to .gguf file")
    args = parser.parse_args()
    
    result = extract_minimal_info(args.filename)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
