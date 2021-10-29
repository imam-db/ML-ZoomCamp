untuk build image
docker build -t titanic-flask .

run docker
docker run -p 6969:6969 titanic-flask

run waitress
waitress-serve --listen=0.0.0.0:6969 predict:app

run docker
docker run -d -p 80:80 docker/getting-started


heroku deploy

heroku login
heroku container:login
cd C:\Users\imamx\Documents\Sementara\Belajar\"Machine Learning ZoomCamp"\Midterm
heroku create -a midtermtitanic
heroku container:push web -a midtermtitanic
heroku container:release web -a midtermtitanic
heroku open -a midtermtitanic