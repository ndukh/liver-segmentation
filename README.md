# Liver segmentation ([CHAOS](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/) challenge).
### An implementation of a CNN model, trained to segment liver on CT-scans.

##### Using: 
```shell
$ python segment.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR]

optional arguments:
        -h, --help            show this help message and exit
        -i INPUT_DIR, --input INPUT_DIR
                        path to the folder with the CT-scans,
                        default: samples/input
        -o OUTPUT_DIR, --output OUTPUT_DIR
                        path to the folder where segmented masks should be saved,
                        default: samples/output

```
##### Clarification:
The exact structure of nested in input folders will be cloned in output.

The program looks for .dcm files in the input folder, repeats their folder
paths in the output folder and saves segmentation masks in .png format, 
following the structure of input folder content.

The work progress is showed by a progress bar.


 