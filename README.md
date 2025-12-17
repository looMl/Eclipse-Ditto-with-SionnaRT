# Integrating SionnaRT with Eclipse Ditto
This demo demonstrates how to configure Eclipse Ditto to update Things and receive notifications via MQTT. The resulting data is fed into SionnaRT for computation and rendering

## Getting Started
### Requirements

1. **Python 3.8-3.11**
	- Recommended: 3.11.9
2. **Docker & Docker Compose**
	- Install from the [official guide](https://docs.docker.com/get-started/get-docker/).
3. **LLVM**
	- Required to run SionnaRT.
	- Download from [LLVM Releases](https://github.com/llvm/llvm-project/releases?q=18.1.8&expanded=true) and ensure it's added to your system PATH.
		- For Windows, select `LLVM-18.1.8-win64.exe` under **Assets**.
4. **Eclipse Ditto**
	- Clone this repository and execute the `docker-compose` file in the `docker` folder.

---
### Installation Steps

1. **Clone the repository**
	```bash
	git clone https://github.com/looMl/Eclipse-Ditto-with-SionnaRT.git
	```
2. **Install dependencies**
Navigate to the `requirements` folder and run the provided `.bat` file to install necessary Python libraries.

3. **Start the Ditto cluster**
- Go to the `docker` folder, open a terminal, and run:
	```bash
	docker-compose up -d
	```
- Verify container health with:
	```bash
	docker ps -a
	```
	Ensure all containers are marked as **healthy**.

4. **Access the Ditto dashboard**
- Open [http://localhost:8080/ui/](http://localhost:8080/ui/) in your browser.

Now we are ready to configure our environment.

---
## Configuration
### 1. Create a Policy
- **Locate `policy.json`**  
Found in the `policies` folder. Contents:
```json
{
  "policyId": "com.sionna:policy",
  "entries": {
    "owner": {
      "subjects": {
        "nginx:ditto": {
          "type": "nginx basic auth user"
        }
      },
      "resources": {
        "thing:/": { "grant": ["READ", "WRITE"], "revoke": [] },
		"policy:/": { "grant": ["READ", "WRITE"], "revoke": [] },
        "message:/": { "grant": ["READ", "WRITE"], "revoke": [] }
      }
    },
	"observer": {
		"subjects": {
			"ditto:observer": {
				"type": "observer user"
			}
		},
		"resources": {
			"thing:/features": { "grant": ["READ"], "revoke": [] },
			"policy:/": { "grant": ["READ"], "revoke": [] },
			"message:/": { "grant": ["READ"], "revoke": [] }
		}
	}
  }
}
```
- **Create the policy**  
Run the following command in the terminal:
	```bash
	curl -X PUT "http://localhost:8080/api/2/policies/com.sionna:policy" -u "ditto:ditto" -H "Content-Type:application/json" -d @policy.json
	```


### 2. Create a Thing
- **Locate `phone.json`**  
	Found in the `things` folder. Contents:
```json
{
	"thingId": "com.sionna:phone",
    "policyId": "com.sionna:policy",
    "attributes": {
        "name": "iPhone"
    },
    "features": {
        "gps": {
            "properties": {
                "position": 0,
				"orientation": 0
            }
        }
    }
}
```
- **Create the thing**  
Run the following command in the terminal:
	```bash
	curl -X PUT "http://localhost:8080/api/2/things/com.sionna:phone" -u "ditto:ditto" -H "Content-Type:application/json" -d @phone.json
	```
### 3. Create Connections
To be able to update things in Ditto and receive notifications from it we need to configure connections. Here we will need two connections:
- source connection for updating thing's state
- target connection for sending notifications about thing's state modifications

To create them, move to the `connections` folder and run:

**Source connection:**
```bash
curl -X POST "http://localhost:8080/devops/piggyback/connectivity?timeout=10" -u "devops:foobar" -H "Content-Type:application/json" -d @source_connection.json
```
**Target connection:**
```bash
curl -X POST "http://localhost:8080/devops/piggyback/connectivity?timeout=10" -u "devops:foobar" -H "Content-Type:application/json" -d @target_connection.json
```

---
 ## Execution 
 Navigate to the `scripts` folder and execute the following scripts in order:
 - First, run `mqtt_subscriber.py`.
 - Then, run `mqtt_publisher.py` and wait for the execution to complete.

After execution, a new folder named `renders` will be created in the same directory. This folder will contain all the generated renderings.

#### Change Scenario to use a custom one:
1. Open your 3D editing software (e.g., Blender).
2. Export the scenario using **Mitsuba**.
3. Replace the meshes and `.xml` files in this repository with your custom files.
