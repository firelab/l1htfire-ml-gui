#L1HTF1re ML spread gui, flask version
Hannah Pinkerton, Missoula Fire Lab, January 2024
Python Version: Python 3.10
Dependencies: flask, tensorflow

To run on local machine, download repository contents, create and activate virtual environment:
`python3.10 -m venv .venv`
`. .venv/bin/activate`

run `flask --app app.py run`

To install using wheel file and waitress:
run `pip install [wheel]` and `pip install waitress`

To launch app:
`waitress-serve --call 'web_app:create_app'`

Note that tensorflow will spit out some warnings to the command prompt; these can safely be ignored.
