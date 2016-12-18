# Machine Learning Engineer Nanodegree
## Capstone Project (West Nile Virus Prediction in Chicago)
Andrew Bauman  
December 13, 2016

## I. Definition

### Project Overview

West Nile virus (WNV), which can result in [West Nile Fever]( https://en.wikipedia.org/wiki/West_Nile_fever) is a bloodborne virus that whose mode of transmission is primarily mosquito to human. According to the World Health Organization [West Nile Virus Fact Sheet]( http://www.who.int/mediacentre/factsheets/fs354/en/)

-	West Nile virus can cause a fatal neurological disease in humans.
-	Approximately 80% of people who are infected will not show symptoms.
-	West Nile virus is mainly transmitted to people through the bites of infected mosquitoes.
-	The virus can cause severe disease and death in horses.
-	Vaccines are available for use in horses but not yet available for people.
-	Birds are the natural hosts of West Nile virus.

#### Transmission and Control Considerations

-	WNV is maintained through a mosquito --> bird --> mosquito cycle where birds are the reservoir hosts
-	The principle vector is considered to be mosquitoes of genus _Culex_ particularly _Cx Pipiens_
-	Birds in North America are particularly susceptible to WNV where infection is highly pathogenic, particularly in crows
-	While birds are a reservoir host, horses are a dead-end host (become infected but don’t spread the infection)
-	WNV outbreaks in animals typically precede human cases so animal health surveillance (bird and horse) are important for early warnings
-	Vector control is achieved through mosquito surveillance and control.  Control  measures consist mainly of source reduction which includes:
    + elimination of mosquito breeding areas through water management (stagnant water is a breeding ground for mosquitoes)
    + administration of chemical and biological agents to destroy mosquito populations
    

#### West Nile Virus in Chicago

The first cases of West Nile in Chicago were reorted in 2002 and by 2004 the City of Chicago and the Chicago Department of Public Health (CDPH) had established a comprehensive survellance and control system.

- Every week from late spring through the fall, mosquitos in traps across the city are tested for the virus - - Results of these tests influence when and where the city will spray airborne pesticides to control adult mosquito populations.
- The associated [Kaggle competition](https://www.kaggle.com/c/predict-west-nile-virus) asks develepers to predict when and where different species of mosquitos will test positive for West Nile virus, in an effort to more effecvitely allocate survellance and control resources.

#### Personal Interest

This competition is particularly interesting to the author of this report due to personal contributions to efforts directed at recommending safe and effective control strategies (spray agents as well as timing and manner of spraying) while a graduate student in **Oregon State University's Department of Environmental and Molecular Toxicology Department**.





### Problem Statement

From the [Kaggle competition description](https://www.kaggle.com/c/predict-west-nile-virus):



> Given weather, location, testing, and spraying data, this competition asks you to predict when and where different species of mosquitos will test positive for West Nile virus. 

More specifically, from the [Data](https://www.kaggle.com/c/predict-west-nile-virus/data) section:

> In this competition, you will be analyzing weather data and GIS data and predicting whether or not West Nile virus is present, for a given time, location, and species. 

The origin of this data is as follows:

- Public health workers set up traps from late May to early October
- From M - W of each week the traps collect mosquitos which are tested for West Nile by the end  of the week
- Test results include [number of mosquitoes, mosquito species, presense or absence of WNV for the cohort]


#### Overview of Data

##### Main Data (Trap Data)

- Each record represents up to 50 tested mosquitos
- Trap locations are described by block number and street name and mapped to Longitude and Latitude
- Satellite traps were used to enhance surveillance.  Each satellite is name by postfixing letters on the parent trap identifier (e.g. T220A is a satellite of T220) 
- Not all locations are tested at all times and records only exist when particular species is found at a certain trap at a certain time.  While predictions will be made on all possible combinations, only actual observations will be scored.


##### Spray Data

- GIS data from 2011 - 2013 mosquito spraying
- Spraying reduces mosquito populatinos and may impact the presence of WNV

##### Weather Data

- NOAA weather data from 2007 - 2014 during test months
- Hot and dry conditions may favor WNV reltative to cold and wet conditions
- Monitoring stations:
    + Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
    + Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level


##### Map Data

- Map data is provided from Open Streetmap
- Map data is primarily for visualization purposes but may also be used in model development


#### Evaluation

From the [submission section of the competition](https://www.kaggle.com/c/predict-west-nile-virus/details/evaluation)

> Submissions are evaluated on area under the ROC curve between the predicted probability that West Nile Virus is present and the observed outcomes.

A receiver operating characteristic (ROC) curve is well explained [here](http://www.dataschool.io/roc-curves-and-auc-explained/).  Essentially it is a plot of True positives as a functino of false positives where each point on the curve is a (x, y) is the (tru pos, false pos) at any given probability cutoff threshold.  That is the threshold in a binary classification where we label a data instance as one class or another.  By integrating the area under the curve (AUC) we have a single number by which we cna compare models.  The larger the AUC the better or model is at discriminating between classes.

#### Process and Submission

- The model will be developed on publin training data and test data both supplied by the host
- The model will be evaluated on private test data supplied by the host, but not revealed to developers

From the submission section of the competition:

> For each record in the test set, you should predict a real-valued probability that WNV is present. The file should contain a header and have the following format:

**Example:**

```
Id,WnvPresent
1,0
2,1
3,0.9
4,0.2
etc.
```

#### Approach to Model Development

This task is well suited to supervised binary classification models, and specifically those models which generate reliable probabilities. Not all suprervised models are amenable to this, Naive Bayes models, I'm looking at you.  I will most likely start with an out of the box ensemble model that scales well, such as XGBoost, but may also try a neural net or or nueral net baesed deep learning model using Keras.

- Prepare Data (wrangling, pre-processing, etc.)
- Explore data
- Perform additional preparation as reveled by exploration (address outliers and missing data, engineer features, transform data, etc.)
- Determine candidate feature set
- Train and Test several models
- Pick a model to move into further development
- Tune model
- Revisit any previous steps to further inform model
- Submit results
- Revisit or repeat any or all of the prior steps until a satisfacotry result has been achieved









### Metrics

Each model will be evaluatied by the AUC of its ROC, as described in the problem statement section. Several other acronyms may be thrown in as part of an intellectual shock and awe campaign.


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
