LOCAL_PYTHON_BIN := python3
VENV_PYTHON_BIN := venv/bin/python
VENV_PYTHON_PIP := venv/bin/pip

venv:
	$(LOCAL_PYTHON_BIN) -m venv ./venv
	. venv/bin/activate
	$(VENV_PYTHON_PIP) install --upgrade pip

freeze:
	$(VENV_PYTHON_PIP) freeze | grep -v "pkg-resources" > requirements.txt

install:
	$(VENV_PYTHON_PIP) install -r requirements.txt

clean:
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.log' -delete

dev:
	FLASK_APP=app.py flask run

run:
	gunicorn -b 0.0.0.0:5124 app:app