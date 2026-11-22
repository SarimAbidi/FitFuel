# FitFuel

**Personalized nutrition and goal-based meal planning**

FitFuel is a Python console application for AL-2001 Programming for AI. It estimates daily calorie and macronutrient targets, searches USDA nutrition data, and builds editable meal plans for weight loss, maintenance, or gain. Users can also look up packaged foods by barcode, classify food descriptions, and track their weight over time.

## How FitFuel works

1. **Set a goal.** Create or reopen a named profile with a weight goal, activity level, diet, and food preference.
2. **Build a plan.** Browse the food database, compare ranked suggestions, or preview a starter plan for breakfast, lunch, dinner, and a snack.
3. **Customize portions.** Exclude specific foods, adjust the calorie share assigned to each meal, and add, remove, or resize individual portions.
4. **Review progress.** Compare the plan against your estimated daily targets. FitFuel saves approved changes and lets you record dated weight check-ins.

## Features and implementation

### Profiles and nutrition targets

`UserProfile` validates profile fields and checks that a target weight matches the selected goal. `UserProfile.targets()` estimates resting energy using the Mifflin–St Jeor equation, applies an activity multiplier, and adjusts the maintenance estimate by up to 10% based on the weight goal. The project allocates 25% of target calories to protein, 45% to carbohydrates, and 30% to fat, then converts those amounts to grams.

### Food database and search

`load_data()` uses pandas to clean **2,646 USDA records** across seven food groups. It keeps **1,938 foods** with complete calorie, protein, carbohydrate, fat, fibre, and sugar values for planning and analysis. `build_catalogue()` joins **130 reviewed annotations** by USDA ID to supply display names, vegetarian eligibility, and starter-meal roles. `search_foods()` matches terms across food names, descriptions, categories, and IDs. Anyone can browse the full database. Vegetarian meal plans use entries with reviewed vegetarian eligibility.

### Food recommendations and starter plans

`GoalBasedRecommender.recommend()` filters reviewed foods by diet and optional category, then ranks them using normalized nutrient values. The score reflects the weight goal and preferences such as higher protein, higher fibre, or lower sugar.

`StarterPlanRecommender.generate()` combines reviewed foods and sample gram portions into candidate meals. It uses scikit-learn’s nearest-neighbor search to find combinations close to each meal’s calorie, protein, carbohydrate, and fat targets. The final selection takes food preferences and ingredient variety into account. The default calorie split is **25% breakfast, 30% lunch, 30% dinner, and 15% snack**. Custom previews can change that split and exclude selected foods before saving.

### Meal editing and progress tracking

`MealPlan.totals()` uses NumPy to scale each food’s per-100-gram nutrition values by the selected portion and sum the results for a meal or the whole day. Portions follow the food’s listed preparation state, such as raw or cooked. `ProfileStore.save()` stores named profiles, plans, and dated weight check-ins in a local SQLite database. Reopening a profile restores its plan. A weight check-in adds to the history, and a profile update changes the weight used for target calculations.

### Barcode lookup and offline demo

`lookup_food()` queries Open Food Facts and validates that the reply matches the requested barcode. It falls back to a cached response, then to a bundled example when one matches the barcode. The lookup displays the source and retrieval time. Missing nutrients appear as unknown. Barcode results are displayed separately from the USDA meal-planning catalogue.

The offline demo uses a fictional profile to generate and reload a meal plan, evaluate the classifiers, produce three charts, and display a bundled barcode response.

### Food description classification and charts

FitFuel checks for a known food name or USDA description before using the text model. `train_models()` cleans descriptions and compares Logistic Regression with Multinomial Naive Bayes using TF-IDF word and two-word features. Both models use the same split with separate training and test food families. `predict_category()` flags unfamiliar text and low-confidence guesses.

`save_charts()` uses Matplotlib to export three charts:

- **Estimated vs. planned calories** compares the daily target with the foods in the current plan.
- **Mean protein by food group** compares USDA category averages per 100 g.
- **Confusion matrix** shows which categories the Logistic Regression model correctly classified or confused on the test set.

## Model evaluation

The dataset contains **2,622 unique cleaned descriptions**, with **2,125 for training** and **497 for testing**. TF-IDF is fitted on the training descriptions. Macro-F1 gives each food category equal weight.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Logistic Regression | 82.29% | 0.796 |
| Multinomial Naive Bayes | 81.09% | 0.766 |

A baseline that always predicts the majority training category scored **26.16% accuracy**. These results measure food-category classification on one held-out split.

## Scope and sources

FitFuel is an educational project. Calorie and macronutrient targets are estimates based on project assumptions. USDA nutrient values depend on food preparation and portion accuracy. The classifier predicts food categories and the vegetarian filter relies on reviewed annotations.

- USDA SR Legacy supplies food descriptions, nutrient values, and source food groups. [USDA SR Legacy dataset](https://agdatacommons.nal.usda.gov/articles/dataset/USDA_National_Nutrient_Database_for_Standard_Reference_Legacy_Release/24661818)
- Open Food Facts supplies the barcode lookup and bundled example. The database is licensed under ODbL 1.0, with individual contents covered by DbCL 1.0. [Open Food Facts](https://world.openfoodfacts.org) · [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) · [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/)
- Mifflin et al. (1990) provide the resting-energy equation. [Original study](https://pubmed.ncbi.nlm.nih.gov/2305711/)

## Project team

| Member | Proposal responsibility |
| --- | --- |
| Sarim Abidi | Integration, console menu, profiles, meal planning, and recommendations |
| Shayan Fatima | Data preparation, NLP, and text classification |
| Hania Waqar | Food API, offline cache, analytics, charts, and verification |

