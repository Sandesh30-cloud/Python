import socket

IP = "0.0.0.0"
PORT = 3000

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((IP, PORT))
    server.listen(5)

    print(f"Server listening on {IP}:{PORT}")

    while True:
        client, addr = server.accept()
        print(f"Client connected: {addr}")

        while True:
            data = client.recv(4096).decode()

            # If client disconnected
            if not data:
                break

            print("Client:", data)

            # If client sent "exit"
            if data.lower() == "exit":
                print("Client requested exit.")
                client.send("Goodbye!".encode())
                client.close()
                break

            # Reply to client
            reply = input("Enter reply: ")
            client.send(reply.encode())

        print("Connection closed.\n")

if __name__ == "__main__":
    main()
