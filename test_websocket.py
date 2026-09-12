import asyncio
import json
import websockets


async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws"

    async with websockets.connect(uri) as websocket:
        print("\nWEBSOCKET TEST")

        query = "What is the status of order ORD0001?"

        print("\nUser:")
        print(query)

        await websocket.send(query)

        response = await websocket.recv()

        print("\nWebSocket Response:")
        print(json.dumps(json.loads(response), indent=2))

        print("\nWebSocket communication successful.")


if __name__ == "__main__":
    asyncio.run(test_websocket())