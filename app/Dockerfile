# This sets up the container with python 3.10 installed.
FROM python:3.10-slim

# This copies everything in your current directory to the /app directory in the contanier.
WORKDIR /app

COPY requirements.txt .
# This runs pip install for all the packages listed in our requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the app files
COPY . .

#This tells Docker to listen on port 80 is the standard port for HTTP
EXPOSE 80

# This command creates a .streamlit directory in the home directory of the contanier.
RUN mkdir -p /.streamlit

#This copies our Streamlit configuration file into the .streamlit directory we just created.
RUN cp config.toml /.streamlit/config.toml

# Similar to the previous step, this copies your streamlit credentials file into the .streamlit directory.
RUN cp credentials.toml /.streamlit/credentials.toml

# This sets the default command for the container to run your app.py script when the container starts.
CMD ["python", "main.py"]