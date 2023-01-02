## Parametric families of deep neural networks with residual adapters [PyTorch + MatConvNet]

## Our Innovations

- To add one more dataset CIFAR10 from https://www.cs.toronto.edu/~kriz/cifar.html, imdbfolder_cifar10 was created. CIFAR10 from the website has different format than the json files in data folder decathlon-1.0 (download_data.sh). New dataloaders were created in imdbfolder_cifar10 w/o using COCO module in imdbfolder_coco. Then the corresponding training files with cifar10 in file names (train_new_task_..._cifar10.py) were created. They were called when the performance of models on CIFAR10 were examed. CIFAR10 seemed to be a easy task that most models, w/ or w/o adapters, gave >98% top-1 accuracy.

- Thanks to Palmetto cluster from Clemson University, we have shown several key points mentioned in the two papers, residual adapters for visual domains, below: 1) learning across domains: teacher models trained on ImageNet can fulfill the image classification tasks of other domains with just adapters learning. 2) ResNet w/ adapters outperformed training from scratch. 3) Parallel adapters could deliver comparable performance as series adapters but with simpler architecture and less computation. 4) Large weight decay was suitable for small datasets with less observations per class.

- Several bugs in the original codes were fixed, including the encoding argument in pickle(), the alpha argument in add(), and the encoding and map_location arguments in torch.load(). The bugs were probably caused by the old version of the original codes.

To practice by yourself, please compare this Readme with the original. Pay attention to the commands 'To train a dataset from scratch/with adapters'. The authors provided the off the shelf models in session of Pretrained networks.


Backbone codes for the papers:
- NIPS 2017: "Learning multiple visual domains with residual adapters", https://papers.nips.cc/paper/6654-learning-multiple-visual-domains-with-residual-adapters.pdf
- CVPR 2018: "Efficient parametrization of multi-domain deep neural networks", https://arxiv.org/pdf/1803.10082.pdf 

Page of our associated **Visual Domain Decathlon challenge** for multi-domain classification: http://www.robots.ox.ac.uk/~vgg/decathlon/

## Abstract 

A practical limitation of deep neural networks is their high degree of specialization to a single task and visual domain.
To overcome this limitation, in these papers we propose to consider instead universal parametric families of neural
networks, which still contain specialized problem-specific models, but differing only by a small number of parameters.
We study different designs for such parametrizations, including
series and parallel residual adapters. We show that, in order to maximize performance, it is necessary
to adapt both shallow and deep layers of a deep network,
but the required changes are very small. We also show that
these universal parametrization are very effective for transfer
learning, where they outperform traditional fine-tuning
techniques.

## Code

### Requirements
- PyTorch
- or MatConvNet with MATLAB

### Launching the code
First download the data with ``download_data.sh /path/to/save/data/``. Please copy ``decathlon_mean_std.pickle`` to the data folder. 

To train a dataset from scratch:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_from_scratch.py --dataset cifar100 --wd3x3 1. --wd 5. --mode bn ``

To train a dataset with parallel adapters put on a pretrained 'off the shelf' deep network:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_adapters.py --dataset cifar100 --wd1x1 1. --wd 5. --mode parallel_adapters --source /path/to/net``
   
To train a dataset with series adapters put on a pretrained deep network (with adapters in it during pretraining):

``CUDA_VISIBLE_DEVICES=2 python train_new_task_adapters.py --dataset cifar100 --wd1x1 1. --wd 5. --mode series_adapters --source /path/to/net``

To train a dataset with series adapters put on a pretrained 'off the shelf' deep network:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_adapters.py --dataset cifar100 --wd1x1 1. --wd 5. --mode series_adapters --source /path/to/net``

To train a dataset with normal finetuning from a pretrained deep network:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_finetuning.py --dataset cifar100  --wd 5. --mode bn --source /path/to/net``

### Pretrained networks
We pretrained networks on ImageNet (with reduced resolution):
- a ResNet 26 inspired from the original ResNet from [He,16]: https://drive.google.com/open?id=1y7gz_9KfjY8O4Ue3yHE7SpwA90Ua1mbR
- the same network with series adapters already in it:https://drive.google.com/open?id=1f1eBQY6eHm616SAt0UXxY9RldNM9XAHb

### Results of the commands above with the pretrained networks
So we train on CIFAR 100 and evaluate on the eval split:

|        |     Val. Acc.     | 
| :------------ | :-------------: | 
| Scratch       |     75.23     |     
| Parallel adapters     |   80.61    |      
| Series adapters       |     80.17      |        
| Series adapters (off the shelf)       |     70.72      |     
| Normal finetuning       |     78.40      |        

## If you consider citing us

For the Visual Domain Decathlon challenge and the series adapters:


        @inproceedings{Rebuffi17,
          author       = "Rebuffi, S-A and Bilen, H. and Vedaldi, A.",
          title        = "Learning multiple visual domains with residual adapters",
          booktitle    = "Advances in Neural Information Processing Systems",
          year         = "2017",
        }


For the parallel adapters:


        @inproceedings{ rebuffi-cvpr2018,
        author = { Sylvestre-Alvise Rebuffi and Hakan Bilen and Andrea Vedaldi },
        title = {Efficient parametrization of multi-domain deep neural networks},
        booktitle = CVPR,
        year = 2018,
        }

