import struct
import pickle
import socket

def receive_all(sock, length):
    data = b''
    while len(data) < length:
        packet = sock.recv(4096)  # Receive in chunks
        if not packet:
            break
        data += packet
    return data

    
def receive_array():
   # host = '10.0.58.25'  # or the IP of the producer if running on a different host
    host='10.7.40.192'
    port = 12345          # Port to connect to
    temp_file = '/tmp/received_array.pkl'  # File to store the received array

    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    buffer = b''  # Buffer for handling excess data

    try:
        while True:
            if len(buffer) < 4:  # If we don't have enough data for the length prefix
                buffer += client_socket.recv(4096)  # Receive more data

            if len(buffer) < 4:  # Still not enough data
                print("No data received, closing connection.")
                break

            # First, read the length of the incoming data from the buffer
            raw_data_len = buffer[:4]
            buffer = buffer[4:]

            # Unpack the length of the data
            data_len = struct.unpack('>I', raw_data_len)[0]  # '>I' means big-endian unsigned int

            # Now, receive the actual data based on the length
            while len(buffer) < data_len:
                buffer += client_socket.recv(4096)  # Keep receiving until we have enough data

            # Extract the actual message from the buffer
            data = buffer[:data_len]
            buffer = buffer[data_len:]  # Save the remaining data in the buffer

            if len(data) == data_len:  # Verify if we received the correct amount of data
                array = pickle.loads(data)
                print("Received array:", array)

                # Store array in a temporary file
                with open(temp_file, 'wb') as f:
                    pickle.dump(array, f)
                print(f"Array saved to {temp_file}")

            else:
                print("Data length mismatch, received", len(data), "bytes, expected", data_len)

    except (ConnectionResetError, OSError) as e:
        print(f"Connection error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        client_socket.close()  # Close the client socket


if __name__ == "__main__":
    receive_array()
