# ML Zoomcamp 5.1 - Intro / Session Overview

## 🧭 Introduction
Welcome to **Week 5** of the **Machine Learning Zoomcamp**!  
This week focuses on **deploying a trained machine learning model** as a **web service** — turning your notebook code into something that other applications can use in production.

Previously, we:
- Trained a **churn prediction model** (Logistic Regression).  
- Evaluated its performance.  

Now, we’ll **take that model**, **save it**, and **serve predictions through a web API**.

## ⚙️ The Deployment Problem

Currently:
- Our model exists **inside a Jupyter Notebook**.  
- It can’t be easily used by other systems.  

We want to:
1. **Save the trained model** to a file.  
2. **Load** that model inside a **web service** (called `churn_service`).  
3. Allow other applications — like a **marketing service** — to send user data and receive predictions.

**Example use case:**
- The marketing service sends a user’s info to the churn service.
- The churn service predicts whether the user is likely to churn.
- If churn risk is high, the marketing service sends a **25% discount** email.

![](./imgs/ml-5-1/1.png)

## 🧩 Focus of This Week
We’ll cover everything needed to turn a notebook-trained model into a **production-ready web service**:

1. **Save and load** models using **Pickle**.  
2. Create a **web service** using **Flask** (Python web framework).  
3. Manage **Python dependencies** with **pipenv** (virtual environment).  
4. Package the service using **Docker** (containerization).  
5. **Deploy** the Docker container to the **cloud** (AWS Elastic Beanstalk).

## 🏗️ Deployment Architecture Overview

| Layer | Description |
|-------|--------------|
| 🧠 **Model** | Trained Logistic Regression churn predictor |
| 🌐 **Web Service (Flask)** | Wraps the model as an HTTP API |
| 📦 **Virtual Environment (pipenv)** | Isolates Python dependencies |
| 🐳 **Docker Container** | Packages the app + dependencies into a reproducible environment |
| ☁️ **Cloud (AWS Elastic Beanstalk)** | Hosts the Docker container as a live, scalable service |

These layers build on top of each other — from **model → service → container → cloud**.

![](./imgs/ml-5-1/2.png)

## 📅 Detailed Plan for the Week

1. **Save & Load Model with Pickle**  
   - Serialize the trained model to disk.  
   - Load it back into memory for use in another process.

2. **Create a Web Service with Flask**  
   - Build an API endpoint to receive user data and return churn predictions.  
   - Understand the basics of HTTP communication between services.

3. **Dependency Management with Pipenv**  
   - Create isolated Python environments for clean project setup.  
   - Manage `Pipfile` and `Pipfile.lock` for reproducibility.

4. **Containerization with Docker**  
   - Package the model, Flask app, and dependencies into a portable container.  
   - Ensure consistency across environments (local, staging, production).

5. **Cloud Deployment with AWS Elastic Beanstalk**  
   - Deploy the Dockerized app to the cloud.  
   - Make it accessible via a public API endpoint.  
   - Learn basic deployment workflows.  
   *(Optional: You can use other platforms instead — AWS is just an example.)*

6. **Explore More Section**  
   - Additional resources for experimenting with other deployment tools or cloud platforms.

---

# ML Zoomcamp 5.2 - Saving and Loading the Model

## 🧭 Introduction
In this lesson, we learn how to:
1. **Save a trained model** from a Jupyter Notebook into a file.  
2. **Load** that model later for inference.  
3. **Convert** the entire notebook into a standalone **Python script** for automated training.

This is a key step in moving from experimentation to **deployable machine learning systems**.

## 🧩 Combining Previous Work
- The starting point is a **single Jupyter Notebook** that contains:
  - All data preparation code from previous weeks.  
  - The `train()` and `predict()` functions.  
  - Model parameters (`C=1`, `5-fold cross-validation`).

- The notebook verifies that the model still performs well using **cross-validation** and **AUC evaluation**.

## 💾 Saving the Model with Pickle
After training, the model still “lives” inside the Jupyter environment — not reusable by other services.  
To make it reusable:
- **Save the model** (and preprocessing objects) to disk using **Pickle**.  
- Store both:
  - The **trained model** (e.g., Logistic Regression).  
  - The **DictVectorizer** (needed to transform inputs before prediction).  

**File naming convention:**  
Include the regularization parameter in the filename (e.g., `model_C=1.bin`).

**Best practice:**  
Use a `with open(...) as f:` statement — ensures the file is properly closed even if an error occurs.

## 🔄 Loading the Model
To simulate another process (like a web service):
- Restart the Jupyter kernel → this clears all variables.
- Import `pickle` and **load** the model and vectorizer from the saved file.
- Once loaded:
  - You can use the vectorizer to **transform new customer data** into a feature matrix.
  - Apply the model’s `predict_proba()` method to get churn probabilities.

**Important:**  
`scikit-learn` must be installed in the environment when loading the Pickle file,  
otherwise the deserialization process will fail.

## 👤 Example Use Case
- Create a new **customer dictionary** (e.g., customer with 1-month tenure).
- Transform it using the vectorizer → get feature matrix.  
- Pass it to the model → get churn probability.  
- The result can inform marketing actions (e.g., send a discount to prevent churn).

## 🧮 Why Convert to a Python Script?
Running everything in Jupyter every time is inefficient.  
Instead, we automate training and saving using a **Python script**:

1. Export the notebook (`.ipynb`) as a **`.py` file**.  
2. Clean up the code:
   - Move parameters (e.g., `C`, file paths) to the top for easy editing.  
   - Group code sections: imports, data prep, training, validation, saving.  
   - Add minimal logging (via `print` statements) to monitor progress.
3. Rename file → `train.py`

OBS: I used the cli for this `jupyter nbconvert 05-deployment-live.ipynb --to python`

This script:
- Trains the model.
- Performs validation.
- Saves the model and vectorizer to disk.

---

# ML Zoomcamp 5.3 - Web Services: Introduction to Flask

## 🧭 Overview
In this lesson, we explore how to make our **machine learning model accessible as a web service** using **Flask**, a lightweight Python web framework.  
The goal is to move beyond running predictions in a local script and instead enable **communication between services** over a network — a key step toward deploying ML in production.

## 🔗 From Script to Web Service
Previously, we used a local `predict.py` script to generate predictions.  
Now, we want to embed this model inside a **churn prediction web service** so that other systems, such as a **marketing service**, can:
- Send user information as a request.
- Receive the churn probability as a response.

This setup allows for **automated decision-making**, such as sending discounts or offers to at-risk customers.

![](./imgs/ml-5-3/1.png)
![](./imgs/ml-5-3/2.png)

## 💡 What is a Web Service?
A **web service** is a method of communication between two devices or systems over a **network** (usually the internet).  
It:
- Receives a **request** (often containing data or parameters).  
- Processes the request.  
- Returns a **response** (such as a prediction, message, or web page).

Web services typically use **HTTP** and methods such as:
- **GET** – retrieve data.  
- **POST** – send data to be processed.  

For example, when using Google Search, the query text (`q=web+service`) is sent as part of a **GET request**, and Google responds with a web page of results.

## ⚙️ Why Use Flask?
**Flask** is a simple and flexible web framework that:
- Handles the technical details of HTTP communication (TCP, routing, etc.).
- Allows developers to turn **Python functions** into **web-accessible endpoints**.
- Makes it easy to create REST APIs for machine learning models.

With Flask, a simple function can be transformed into a web route that can respond to requests like:
- `GET /ping` → returns a simple response (e.g., “pong”).

## 🧩 Creating a Basic Web Service
To illustrate how Flask works, we begin by:
1. Creating a simple **function** that returns a fixed response.  
2. Using Flask to expose this function as a **web endpoint**.  
3. Running the Flask application on a **local server** (using a specified port).  
4. Accessing it through:
   - The **command line** (with `curl`).  
   - A **web browser** (via `http://localhost:9696/ping`).

When accessed, the service responds with a message (e.g., “pong”), confirming successful communication.

## 🌍 Interacting with the Web Service
- **From Terminal:** Use command-line tools (like `curl`) to send HTTP requests directly to the running Flask service.  
- **From Browser:** Enter the service URL to trigger the request visually and receive the response in the browser.  
- Flask automatically logs each request, showing the endpoints being accessed and their responses.

## ✅ Summary
In this lesson, we learned:
- What **web services** are and why they’re essential for model deployment.  
- How to use **Flask** to turn a Python function into a simple, callable web service.  
- How to **communicate** with that service from a terminal or browser.  

This provides the foundation for building more advanced APIs that expose ML models to other systems.

---

# ML Zoomcamp 5.4 - Serving the Churn Model with Flask

## 🧭 Overview
In this lesson, we focus on transforming the **churn prediction model** into a fully functional **web service**.  
This builds upon the previous session, where we learned how to create simple Flask applications.  
Now, the objective is to expose the trained model so that other systems can send **customer data** and receive **churn probability predictions** over the network.

![](./imgs/ml-5-4/1.png)

## ⚙️ From Model to Web Service
The **goal** is to take the churn prediction logic—previously executed locally in a script—and embed it into a **Flask-based web service**.  
This service acts as a bridge between:
- The **trained machine learning model** (for inference).  
- **External clients** (such as marketing or analytics systems) that need to query predictions.

The process involves:
1. **Loading the trained model and vectorizer** (saved with Pickle).  
2. **Creating API endpoints** to handle requests and responses.  
3. **Returning model predictions** as structured JSON output.

## 📦 Input and Output via JSON
To enable smooth communication, the service uses **JSON** as the data exchange format:
- **Input JSON:** Contains user or customer information (features for prediction).  
- **Output JSON:** Returns the **predicted churn probability** as a numeric value.

This standardized format allows integration with many other tools and programming languages.

## 🧩 Running the Flask App in Debug Mode
During development, the service runs in **debug mode**, enabling:
- Automatic reload on code changes.  
- Detailed error reports when something goes wrong.  

Common issues during this phase include **HTTP 500 Internal Server Errors**, which typically result from missing imports, incorrect data formatting, or unavailable model files.

## 🌐 Communicating with the Churn Service
Once the Flask app is running, it can be tested using:
- **Browser access** (via a GET request to localhost).  
- **Command-line tools** like `curl` or `httpie` for POST requests.  

If errors appear (e.g., connection issues or malformed input), Flask logs them directly in the terminal, making debugging straightforward.

## 🚀 Production Deployment with Gunicorn
After validating the service locally, the next step is to prepare it for **production use**.  
Instead of relying on Flask’s built-in server (meant for development only), a **production-grade WSGI server** like **Gunicorn** is used.

**Gunicorn** provides:
- Better concurrency and performance.  
- Stability under higher traffic loads.  
- Compatibility with containerized or cloud deployments (e.g., Docker + AWS).

Running the model with Gunicorn ensures that the web service can handle multiple prediction requests efficiently and securely.

## ✅ Summary
In this lesson, we covered:
- How to **integrate a trained churn prediction model** into a **Flask web service**.  
- How to use **JSON** for data communication between client and server.  
- How to **test** and **debug** the service locally.  
- How to prepare the app for **production deployment** using **Gunicorn**.

This represents a major milestone — transforming a local machine learning model into a **deployable, network-accessible prediction API**.

---

# ML Zoomcamp 5.5 - Python Virtual Environment: Pipenv

## 🧭 Overview
In this lesson, we learn about **virtual environments** — isolated spaces that allow different projects to use different library versions without conflicts.  
We also explore **Pipenv**, a tool for managing dependencies and virtual environments in Python projects.

The goal is to ensure that multiple machine learning services (like *Churn Service* and *Lead Scoring Service*) can coexist on the same machine, even if they depend on different versions of the same libraries.

![](./imgs/ml-5-5/1.png)
![](./imgs/ml-5-5/2.png)
![](./imgs/ml-5-5/3.png)

## 🧩 The Problem: Dependency Conflicts
When using Python globally (via `pip install`), libraries are installed system-wide.  
This creates problems when:
- Two projects require **different versions** of the same library.  
- Upgrading or removing one dependency **breaks another project**.

For example:
- The **Churn Service** might use `scikit-learn==0.24.2`.  
- The **Lead Scoring Service** might need `scikit-learn==1.0`.  

Without isolation, these two versions cannot coexist, leading to **version conflicts**.

## 🧱 The Solution: Virtual Environments
A **virtual environment** solves this by creating an isolated folder containing:
- Its own **Python interpreter**.
- Its own **installed libraries**.

Each project then runs independently:
- The Churn Service has its own Python and dependencies.  
- The Lead Scoring Service has its own, separate setup.  

This ensures that installing or updating libraries in one project does not affect others.

## 🧰 Tools for Virtual Environments
Python offers several tools for managing virtual environments:
- **`venv`** – the built-in Python module.  
- **Conda** – used in the Anaconda ecosystem.  
- **Pipenv** – an officially recommended tool for dependency management.  
- **Poetry** – a modern alternative with similar features.

In this course, we focus on **Pipenv**.

## ⚙️ What is Pipenv?
**Pipenv** simplifies both **dependency management** and **environment creation**:
- Replaces manual use of `pip install`.  
- Keeps dependencies tracked in configuration files.  
- Ensures reproducibility across machines.

It automatically creates:
- A `Pipfile` – lists project dependencies.  
- A `Pipfile.lock` – records exact versions for reproducibility.

## 📦 Installing and Using Pipenv
After installing Pipenv, dependencies are added using commands like:
- Installing libraries (e.g., `scikit-learn`, `flask`, `numpy`).  
- Specifying exact versions to avoid warnings or compatibility issues.  
- Creating development vs. production dependencies (e.g., `dev-packages` section).

Once installed, Pipenv generates:
1. **`Pipfile`** – declares the project dependencies.  
2. **`Pipfile.lock`** – locks the versions for deterministic builds.

## 🧾 Understanding Pipfile and Pipfile.lock
- **`Pipfile`** specifies the high-level dependencies (e.g., Flask, NumPy).  
- **`Pipfile.lock`** stores the **exact version** and **checksum** of every package.  

This ensures that when another developer clones the repository and runs `pipenv install`,  
they get the exact same environment as the original author.

This guarantees **reproducibility** and **consistency** across different machines.

## 🧠 Running Code in Pipenv
Pipenv provides isolated execution environments through commands such as:
- **`pipenv shell`** – opens a shell session inside the project’s virtual environment.  
- **`pipenv run`** – runs a command (like `python` or `gunicorn`) directly inside the environment.  

This ensures the project uses the **correct Python interpreter and dependencies** every time.

## 🔒 Why Isolation Matters
Virtual environments isolate **Python-level dependencies**,  
but they **don’t isolate system-level libraries** (e.g., OpenMP, libc, etc.).  

If two services require different system libraries, virtual environments alone can’t resolve that.  
For such full isolation (including OS-level dependencies), we use **Docker**,  
which provides complete containerization.

## ✅ Summary
In this lesson, we covered:
- The **problem** of dependency conflicts between multiple projects.  
- How **virtual environments** isolate Python dependencies.  
- How **Pipenv** helps manage and reproduce these environments.  
- The structure and role of **Pipfile** and **Pipfile.lock**.  
- How to run Python services inside Pipenv environments.  

---

# ML Zoomcamp 5.6 - Environment Management: Docker

## 🧭 Overview
In this lesson, we explore **Docker**, a tool for creating isolated environments that go beyond Python’s virtual environments.  
While tools like Pipenv isolate Python dependencies, **Docker isolates the entire system environment** — including the operating system, Python version, and system libraries.

Docker enables us to package machine learning services (like the **Churn Service** and **Lead Scoring Service**) into independent, self-contained **containers** that can run reliably on any system.

![](./imgs/ml-5-6/1.png)
![](./imgs/ml-5-6/2.png)

## 🧱 From Virtual Environments to Containers
Previously, we used **virtual environments** to separate Python dependencies between projects.  
However, virtual environments share the same operating system, which means:
- System-level dependencies can still conflict.
- Services might need different OS versions or compilers.

**Docker** solves this by allowing each service to live in its **own container**, isolated from everything else on the host machine.

Each container includes:
- Its own **OS environment** (e.g., Ubuntu 18.04, Ubuntu 20.04, Amazon Linux).  
- Its own **Python version** and **dependencies**.  
- Any required **system tools** (like GCC or OpenMP).

From inside a container, each service behaves as if it’s the only program running on the machine.

## 🧩 Complete Isolation Across Services
Example setup:
- **Churn Service** → uses Ubuntu 18.04 with `scikit-learn==0.24`.  
- **Lead Scoring Service** → uses Ubuntu 20.04 with `scikit-learn==1.0`.  
- **Email Service** → runs on Amazon Linux with mail-sending tools.

All these containers can run simultaneously on the same host (e.g., your laptop or a cloud server), completely **independent and conflict-free**.

## ☁️ Why Docker Matters
The major advantage of Docker is **portability**:  
Once a container is built, it includes everything needed to run the service.  
You can easily move it:
- From your laptop → to a server.  
- From local development → to a cloud environment.

No additional setup is needed — the container “just works.”

## 🧠 Key Concepts in Docker

### 🏗️ **Docker Image**
A **Docker image** is a blueprint that defines:
- The base operating system (e.g., Debian or Ubuntu).  
- The required Python version and dependencies.  
- Any code or files needed for the service.  

### 📦 **Docker Container**
A **container** is a running instance of an image.  
You can have multiple containers running from the same image, each isolated from one another.

### 📜 **Dockerfile**
The **Dockerfile** describes the instructions to build the image, such as:
1. **Base image** – e.g., `python:3.8-slim`.  
2. **Copying project files** (model, app, dependencies).  
3. **Installing dependencies** from `Pipfile` and `Pipfile.lock`.  
4. **Exposing ports** to allow network access.  
5. **Setting the entry point** to start the service automatically (e.g., using Gunicorn).

## ⚙️ Building and Running Containers
Workflow summary:
1. Choose a base image (e.g., Python 3.8-slim).  
2. Create a `Dockerfile` that defines installation and runtime steps.  
3. **Build the image** using `docker build`.  
4. **Run the container** using `docker run`.  
5. **Map ports** between the container and host machine (e.g., map `9696` inside the container to `9696` on the host).

This allows external scripts (like `predict_test.py`) to communicate with the web service running inside the container.

## 🌐 Networking in Docker
Each container runs in its own network space.  
To allow outside communication:
- **Expose** the port inside the container (e.g., `EXPOSE 9696`).  
- **Map** it to a port on the host machine (e.g., `-p 9696:9696`).  

This enables local applications or other containers to send requests to the web service inside Docker.

## 🧰 Example: Packaging the Churn Service
In the **Churn Prediction project**, Docker is used to:
1. Bundle the trained model and Flask API (`predict.py`).  
2. Include all dependencies and Python packages.  
3. Run the service using **Gunicorn** inside the container.  
4. Expose port `9696` for communication.  

After building and running the image, the model can receive prediction requests from any external client — all within the containerized environment.

## ✅ Summary
In this lesson, we covered:
- What **Docker** is and how it differs from virtual environments.  
- How containers provide **complete isolation** for services.  
- How to build and run containers using **Dockerfiles**.  
- The concept of **port mapping** for communication.  
- How to package and run the **churn prediction service** inside Docker.

Docker allows us to build reliable, portable, and isolated machine learning services that can be easily deployed anywhere.

---

# ML Zoomcamp 5.7 - Deployment To The Cloud: AWS Elastic Beanstalk (Optional)

### 🧭 Overview
In this final practical lesson of Session 5, we learn how to **deploy a Dockerized machine learning service to the cloud** using **AWS Elastic Beanstalk (EB)**.  
The goal is to make our **churn prediction model** accessible via the internet, running inside a managed cloud environment that can automatically scale.

This lesson builds on the previous one, where we containerized the churn prediction service using Docker.

![](./imgs/ml-5-7/1.png)

## 🏗️ From Local to Cloud Deployment
Previously:
- We built a Docker image containing our **Flask web service** and trained **ML model**.
- We tested the service **locally** on port `9696`.

Now:
- We take that Docker image and **deploy it to AWS**,  
  turning it into a publicly accessible web service.

Although the lesson uses AWS, you can deploy the same container using:
- **Google Cloud Run**
- **Azure App Service**
- **Heroku**
- **PythonAnywhere**, or any other platform that supports Docker.

## 🔑 Setting Up AWS
To deploy on AWS:
1. You need an **AWS account** (free-tier works for testing).  
2. Configure your account using **access keys** for programmatic access.  
3. Optionally, follow the detailed setup guide available on the course website.

In this demo, the instructor uses an **EC2 instance** running on AWS to build and deploy Docker containers remotely.  
This approach avoids hardware compatibility issues (e.g., ARM vs. x86 architectures) and allows faster cloud-based builds.

## 🚀 Introducing Elastic Beanstalk (EB)
**Elastic Beanstalk** is a managed AWS service that automates:
- Deployment of Docker containers or applications.  
- Scaling up and down based on traffic.  
- Load balancing between multiple container instances.  

It simplifies running web services in the cloud without manual infrastructure setup.

### 🧩 How It Works
1. Your **Docker container** (the churn service) runs inside Elastic Beanstalk.  
2. The **marketing service** or other clients send HTTP requests to the EB endpoint.  
3. EB forwards requests to one or more running containers.  
4. When traffic increases, EB automatically **scales up** (adds more containers).  
5. When traffic decreases, EB **scales down** to save resources.

This elasticity allows cost-efficient, production-grade deployments.

## ⚙️ Installing the Elastic Beanstalk CLI
To deploy via the command line:
1. Install the **EB CLI** (Elastic Beanstalk Command-Line Interface).  
2. Add it as a **development dependency** in the Pipenv environment.  
3. Use it from within the virtual environment (`pipenv shell`).

This tool handles:
- Application setup (`eb init`)  
- Environment creation (`eb create`)  
- Local testing (`eb local run`)  
- Termination of cloud resources (`eb terminate`)

## 🧠 Deployment Workflow

### **Step 1: Initialize the Application**
Run the initialization command to tell EB that the project uses **Docker**.  
You’ll specify:
- Platform: Docker  
- Region: closest AWS region (e.g., `eu-west-1` for Ireland)

This generates a configuration folder called `.elasticbeanstalk`,  
which stores deployment settings such as platform and region.

### **Step 2: Test Locally**
Before deploying, you can test the setup locally:
- Build and run the container with `eb local run`.  
- Test it via the existing `predict_test.py` script to ensure predictions work.  

### **Step 3: Deploy to AWS**
Run `eb create churn-serving` to:
- Upload the Docker image and configurations.  
- Launch a cloud environment running the service.  
- Automatically configure load balancing and auto-scaling.

Once complete, AWS provides a **public URL** for your service (e.g.,  
`http://churn-serving-env.elasticbeanstalk.com`).

### **Step 4: Test the Cloud Service**
Update the request script:
- Replace `localhost` with the public AWS host.
- Remove the port number (Elastic Beanstalk maps port `9696` to port `80` by default).

Run the test — you’ll receive real-time predictions from the service hosted in the cloud!

## 🔒 Security Considerations
By default, the Elastic Beanstalk service is **publicly accessible**.  
Anyone with the URL can send requests to your model.

For production systems:
- Restrict access to specific IPs or internal services.
- Use AWS security groups or private networks (VPCs).
- Consider adding **authentication** or **API gateways**.

For educational or testing purposes, public access is fine — just remember to **terminate** the service when finished.

## 🧹 Cleaning Up
When you’re done testing:
- Terminate the Elastic Beanstalk environment using `eb terminate churn-serving`.  
- This stops billing and removes all deployed resources.  
- You can also do this from the AWS Management Console.

## ✅ Summary
In this lesson, we learned how to:
- Deploy a **Dockerized ML model** to the **cloud** using AWS Elastic Beanstalk.  
- Use the **EB CLI** to initialize, run, and terminate environments.  
- Test the service both locally and on the cloud.  
- Understand **auto-scaling**, **load balancing**, and **security** basics.

This makes our churn prediction model a **fully deployed web service**, accessible and scalable in the cloud.