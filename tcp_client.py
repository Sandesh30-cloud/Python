import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 3000))

# Send first message
client.send("Hello server!".encode())

# Receive first reply from server
response = client.recv(4096).decode()
print("Server:", response)

while True:
    # Get user input
    message = input("Enter message to send to server (type 'exit' to quit): ")

    # Send message to server
    client.send(message.encode())

    # If user typed exit → stop sending further messages
    if message.lower() == "exit":
        print("Closing connection...")
        break

    # Receive next server reply
    reply = client.recv(4096).decode()
    print("Server:", reply)

client.close()
