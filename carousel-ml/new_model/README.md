# Carousel ML Model Gui  
Not hosted on the server, uses most recent model with linear outputs  
See line 75 in newest_caro.py for how the graph is getting the predictions  
Hannah Pinkerton, Missoula Fire Lab, Winter 2024  
Python Version: 3.10  
Dependencies: see requirements.txt  

ON LINUX:
To run on local machine, download contents of repository, create and activate python virtual environment  

`python3.10 -m venv .venv`  
`. .venv/bin/activate`

Use pip to install dependences:  
`pip install -r requirements.txt`  

Run app using streamlit command:  
`streamlit run newest_caro.py`  

To run using docker file, ensure docker is installed.  
If not: https://docs.docker.com/engine/install/#server  

To build app using docker run:  
`docker build -t carousel_ml_new .`

Check installation with:  
`docker images`  

You should see a repo named carousel_ml  

Then run app:  
`docker run carousel_ml_new`
