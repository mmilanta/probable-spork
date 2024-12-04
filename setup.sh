rm -r ./venv
python -m venv venv                 
. venv/bin/activate                                                                      
pip install -r requirements.in
pip freeze > requirements.txt