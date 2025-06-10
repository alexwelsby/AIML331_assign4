# Assignment 3 for AIML331: Computer Vision.

(The repo name is incorrect.)

## Part 1: CNN Model Implementation and Analysis 

Part 1's [code](https://github.com/alexwelsby/AIML331_assign4/blob/master/Assignment3-Part1.ipynb) can be opened in Jupyter Notebook.

Run All Cells to run.

It will take a significant amount of time to run all CNNs given the number of parameters (see run_model()). To account for the likelihood of needing to interrupt training, I added the methods **append_text_to_file** and **extract_models_from_file** so the algorithm will 'save' the results of each model (and later 'load' and check which have already been run to avoid running them again).

**class configConvNet(nn.Module)**: The CNN class.

**extract_models_from_file(filepath)**: Fetches the models that have already been run from the saved AllAccuracies.txt file and returns them as an array. Helps run_model() skip models that it's already run, in case running is interrupted.

**append_text_to_file(file_path, text_to_append)**: Called at the end of **test_model()**. Used to append the current model's parameters and accuracy to AllAccuracies.txt so it can later be retrieved by **extract_models_from_file()**.

**generate_charts(num_layers, activation, batch_norm, residuals, train_losses, val_accuracies)**: Called at the end of run_model. Saves training loss and validation curve charts to .pngs in the root directory. A model's respective parameters are saved in both the filename and the chart's title for easy access.

**run_model()**: Runs a series of 48 CNN models using the following parameters: 

```
batch_normalisation = [True, False]

residuals = [True, False]

num_layers = [3, 4, 5, 6 ] 

activation_funcs = ["ReLU", "LeakyReLU", "GELU"]
```
Validates the model, sends validation information to **generate_charts()**. Finally, calls **test_model()**.

**test_model(model, descript, batch_size, device)**: Tests the model. Calls **append_text_to_file(file_path, text_to_append)** with the model's parameters.
