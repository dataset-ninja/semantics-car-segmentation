Dataset **Car Segmentation** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzI1MDRfQ2FyIFNlZ21lbnRhdGlvbi9jYXItc2VnbWVudGF0aW9uLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogIkg4czh1ZnJMZUw3ajBtMFJPQ0pxeXhVZCt2aWU4eXZ0bkFTOWlYOHRLWUE9In0=)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Car Segmentation', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

