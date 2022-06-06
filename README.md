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

* DATAROBOT_ENDPOINT=https://app2.datarobot.com/api/v2
* DATAROBOT_API_TOKEN=
* DR_TRAIN_DATASET_FILEPATH=
* DR_TEST_DATASET_FILEPATH=

As shown above, the default endpoint root for the DataRobot AI Platform Trial and Self-Service users is https://app2.datarobot.com.

5. Once the project is configured, the below demonstrates the output of each of the primary function calls incorporated into the main() function of datarobot_demo.py.

    ### load_environment_and_client() ###
    ```
    def load_environment_and_client():
    load_dotenv('.env')
    dr.Client()
    return
    ```

    Very simple, no output generated.

    ### train_dataset, test_dataset = load_datasets() ###
    ```
    train_dataset_file_path = os.getenv('DR_TRAIN_DATASET_FILEPATH')
    test_dataset_file_path = os.getenv('DR_TEST_DATASET_FILEPATH')

    train_dataset = dr.Dataset.create_from_file(train_dataset_file_path)
    test_dataset = dr.Dataset.create_from_file(test_dataset_file_path)
    ```

    Load the train and test datasets into file from the location specified in the .env file, and return the datasets in-memory. Note the alternative approach to load from a Pandas DataFrame included in the datarobot_demo.py file.
    
    ### project = create_project_and_target(train_dataset) ###
    ```
    project = dr.Project.create_from_dataset(train_dataset.id, project_name=f'Prospect_Sentiment')

    project.set_target('signed_contract', worker_count=-1)
    ```

    Create a project from the training_dataset and set the target for DataRobot to begin EDA.

    ### explore_training_dataset_features(train_dataset, histogram=True) ###
    ```

    ```
    
    ### model = train_model(project) ### 
    get_top_of_leaderboard(project, verbose=True)
    predictions = predict_against_model(project, model, test_dataset)

