FROM python
RUN apt-get update
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
# RUN python manage.py migreate
# CMD python manage.py runserver