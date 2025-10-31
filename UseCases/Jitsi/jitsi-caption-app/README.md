# How to run the project on localhost
In the project directory, you can run:

`npm start`\
`npm run start-api`

You might also need to pip install flask, flask_cors and potentially other python requirements.

# After launching the app
1. From our app you can only join a meeting (you cannot create one). To create a meeting, you have to first create an account with jitsi meet through their wbesite and host a meeting. Visit [meet.jit.si](meet.jit.si)

2. When creating the meeting, use this code: `Ad=_123rsA` (you could use arbitrary code, but this one is hard coded in App.js for now)

3. Now you should be able to login from this app hosted at localhost:3000 




# TODO
- figure out a way to stream the video without displaying it (should be easy) 
- test if the frames actually do receive on backend (by e.g. saving some of them or smth like that) DONE
- implement the model inference into the backend (by copying our existing code from GESTURE V1) DONE
- instead of sending incrementing counter back to front end, send the actual prediction DONE
- list the requirements.py
- style the website
- add .flaskenv to .gitignore
- optional: deploy / allow login from our link (now user can only join existing meeting), ...
