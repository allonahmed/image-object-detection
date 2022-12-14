# Detecting clothing objects in images

### Project Details: 
For this project, we want to be able to detech clothing types from an image such as shirts and pants. This will help in grouping clothing into categories for organizing a database involving that contains images of clothes. Further down the line we will need to also distinguish different parts of clothing, for example, in a sneaker, we would want to identify the toebox or tongue and create new images of the item.

### Example using ImageAI:
<div>
  <image src='https://i.imgur.com/3wInz9b.png'/>
</div>  

### Setup:
1. Install the current release of CPU-only TensorFlow: ```conda create -n tf tensorflow```
2. Activate the conda enviroment you just created: ```conda activate tf```
3. Install additional dependencies: ```conda -c conda-forge opencv imageai keras```
4. Add file you want to test in the input folder and run: ```python3 main.py```
5. View your results!
