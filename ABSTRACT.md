**Car Segmentation** dataset encompasses 211 car side view images, meticulously annotated into four distinct classes: *car*, *wheel*, *lights*, and *window*. The dataset creation process involved a comprehensive approach, including image gathering through various methods and detailed mask annotation using [VoTT](https://github.com/microsoft/VoTT), ultimately culminating in a valuable resource for training and developing object recognition models. For those interested in utilizing the dataset without manual annotation, a pre-annotated dataset has been thoughtfully provided by the creator for use in training models.

Firstly, the dataset creator needs to gather a substantial collection of car side view images. Multiple methods can be utilized for this purpose:

Unsplash: Unsplash is a valuable resource that offers a wide range of images with permissive copyright terms. While it's a great platform for acquiring images, it's worth noting that there may be limitations in finding images specifically depicting car side views.

Google Images: Utilizing extensions available in web browsers like Google Chrome and Firefox, one can conveniently download all the images from the active tab. This is particularly useful when searching for images using specific keywords like "car side," with the option to exclude undesired results using the "-" symbol.

Capturing Images in the Field: If the dataset creator possesses a suitable camera, they have the option to capture images themselves by venturing outdoors. However, it's essential to be mindful of privacy and copyright concerns, especially when planning to share the dataset publicly. Blurring people's faces and license plates is a prudent measure in this case.

Approximately 200 images may be needed to compile a comprehensive dataset that yields satisfactory results.

Once the images are gathered, the next step involves creating mask annotations. A mask annotation is essentially an image where pixel values are assigned to represent different classes. These annotations can take various formats, such as black-and-white PNG, colorful PNG, or COCO-style JSON, among others.

For the purpose of annotation, numerous tools are available. In this case, the dataset creator chose to employ VoTT, a tool that facilitates the creation of bounding boxes or polygons for image segmentation. While some models may not directly comprehend polygons, there are software solutions, such as Intelec AI, capable of translating them into usable masks.

To begin the annotation process with VoTT, one must set up a project, specifying the location for saving the annotated images as the "Target connection" and the dataset location as the "Source connection." The polygon annotation option can be selected, and annotations can be made by drawing polygons around the objects of interest. After drawing a polygon, labels are assigned to the respective objects.

Once the annotation process is complete, it's crucial to perform an additional step. In the left panel, export settings should be configured to export in the VoTT format. Then, in the top panel, the export function is initiated, resulting in the creation of a significant JSON file (e.g., [project-name]-export.json). This file is of utmost importance as it contains all the annotations in a format that can be understood and processed by Intelec AI.

In this particular case, the dataset creator chose to categorize cars into four distinct classes: the car itself (for determining its location in the image), the wheels, the front and back lights (labeled as "lights"), and the windows, including rear windows and windshields (labeled as "window").

For those interested in utilizing the dataset without the need for manual annotation, the creator kindly provided a pre-annotated dataset for use in training models.
