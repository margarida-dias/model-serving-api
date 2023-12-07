
if __name__ == "__main__":

    import requests

    inputs = [[1237, 100.0, 5.0], [1236, 80.0, 3.0], [1235, 60.0, 20.0], [1234, 40.0, 20.0]]

    inference_request = {
        "inputs": [
            {
                "name": "echo_request",
                "shape": [len(inputs), 3],
                "datatype": "INT32",
                "data": inputs,
            }
        ]
    }

    # endpoint for local development
    local_endpoint = "http://localhost:8080/v2/models/model2/infer"
    response = requests.post(local_endpoint, json=inference_request)

    print(response.json())

    local_endpoint = "http://localhost:8080/v2/models/model-serving-api/infer"
