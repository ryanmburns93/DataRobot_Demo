# AutoML with DataRobot Demo
This project Explore MLOps and AI-powered application development with DataRobot.

## Automated Machine Learning Overview ##
Automated machine learning (AutoML) is "the process of automating the time-consuming, iterative tasks of machine learning model development," according to [Microsoft](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml). AutoML technologies have matured significantly in the last five years and are becoming increasingly common-place in the mature data scientist's toolkit. Recent research by intelligence firm [ReportLinker.com](https://www.globenewswire.com/news-release/2022/05/19/2446648/0/en/Global-Automated-Machine-Learning-Market-Growth-Trends-COVID-19-Impact-and-Forecasts-2022-2027.html) describes the AutoML market as highly competitive and moderately fragmented, composed of several embedded market players joined by recent entrants seeking to capitalize on the advancement of data science technologies and capabilities which are organically raising the value and importance of AutoML tools to competitive business.

According to [AIMultiple](https://aimultiple.com/automl-software), the top AutoML solution vendors with English support include:
* [Dataiku](https://www.dataiku.com/)
* [DataRobot](https://www.datarobot.com/)
* [H2O](https://h2o.ai/)
* [Akkio](https://www.akkio.com/)
* [dotData](https://dotdata.com/)

## DataRobot AutoML Tutorial ##
This tutorial provides a simple demonstration of incorporating autoML technology into a data science pipeline. The project is built using DataRobot and requires a registered account to complete.

The final project will harness (anonymized for this demonstration) daily message data for applying sentiment analysis to the messages for ranking the follow-up outreach conducted the following day to prioritize prospects with greater likelihood of contracting with the business.

The tutorial is written for use on Windows 10 using Powershell and Python in VSCode. The Python installation is Python 3.10.4.

### Getting Started ###
1. Create a new GitHub repository. Create a virtual environment for installing libraries for the project by navigating to the repository and calling the below command.
```
cd ./DataRobot_Application/
python -m venv .venv
./.venv/Scripts/Activate.ps1
```

2. Clone this GitHub repository using the code below. Then create a .env file for storing your own secret keys. Make sure you have your own .gitignore file and it includes an exclusion for your .env file.
```
git clone https://github.com/ryanmburns93/DataRobot_Application.git
New-Item -Path . -Name ".env" -ItemType "file"
```

3. Install the required dependencies from requirements.txt into the virtual environment.
```
pip install -r requirements.txt
```

4. Add your project details to your .env file. Required values are:
```
DATAROBOT_ENDPOINT=https://app2.datarobot.com/api/v2
DATAROBOT_API_TOKEN=<your token here>
DR_TRAIN_DATASET_FILEPATH=<your train dataset filepath here>
DR_TEST_DATASET_FILEPATH=<your test dataset filepath here>
```
As shown above, the default endpoint root for the DataRobot AI Platform Trial and Self-Service users is https://app2.datarobot.com.

5. Once the project is configured, the functions in datarobot_demo.py are written and called linearly with significant comments. Further details and a walkthrough demonstrating execution of the project are available [here]().
