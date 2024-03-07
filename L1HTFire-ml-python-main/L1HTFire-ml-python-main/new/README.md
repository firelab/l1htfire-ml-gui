# L1HTFire-ML Gui, Python streamlit version  
Hannah Pinkerton, Missoula Fire Lab, February 2024  
Python Version: 3.10  
Dependencies: streamlit, tensorflow, pandas, numpy, plotly  

On Linux, to run on local machine, download contents of repository, create and activate python virtual environment:

`python3.10 -m venv .venv`

`. .venv/bin/activate
`

Use pip to install dependencies:

`pip install -r requirements.txt`

Run app using streamlit command:

`streamlit run app.py`

To use docker file, ensure docker is installed and run:

`docker build -t streamlit_app .`

Check installation with:

`docker images`

You should see a repository named streamlit_app. Then to run app:

`docker run streamlit_app`

To run the app on the production server:

    streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false --server.enableWebsocketCompression=false --server.port 8501
