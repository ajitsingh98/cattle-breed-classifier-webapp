# Heroku Web App Deployment

[Heroku](https://www.heroku.com/) was utilized to deploy the web app.

### Input to Heroku App

This output file, **`model.pkl`** is the input to the Heroku app.  

 
## Heroku Setup
If you don't have a Heroku account, create one here: [www.heroku.com](https://www.heroku.com/). 
For deploying this project, I have used following steps.
* Upload your code to any github repository and make sure that it is running on local server.
* Make sure you have same hierarchy of folders and files as given in this repository.
* Create a <a href="https://dashboard.heroku.com/new-app">heroku app</a> and name it according to your choice.
* Go to the deployment method and then use github method for deployment.
* It will deploy your model within a couple of minutes.
* See the *View logs* option for more details and working of your deployed model.


Note:  After 15 minutes of inactivity, Heroku will suspend the app.  The next time the web app is called, Heroku will restart the app.  There could be a slight delay in starting the app.
 
