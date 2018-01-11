# dl4clj

Port of [deeplearning4j](https://github.com/deeplearning4j/) to clojure

# Contact info
If you have any questions,
- my email is will@yetanalytics.com
- I'm will_hoyt in the clojurians slack
- twitter is @FeLungz (don't check very often)

## TODO
- update examples dir
- finish README
  - add in examples using Transfer Learning
- finish tests
  - eval is missing regression tests, roc tests
  - nn-test is missing regression tests
  - spark tests need to be redone
  - need dl4clj.core tests
- revist spark for updates
- write specs for user facing functions
  - this is very important, match isnt strict for maps
  - provides 100% certianty of the input -> output flow
  - check the args as they come in, dispatch once I know its safe, test the pure output
- collapse overlapping api namespaces
- add to core use case flows

## Features

### Stable Features with tests

- Neural Networks DSL
- Early Stopping Training
- Transfer Learning
- Evaluation
- Data import

### Features being worked on for 0.1.0

- Clustering (testing in progress)
- Spark (currently being refactored)
- Front End (maybe current release, maybe future release. Not sure yet)
- Version of dl4j is 0.0.8 in this project.  Current dl4j version is 0.0.9
- Parallelism
- Kafka support
- Other items mentioned in TODO

### Features being worked on for future releases

- NLP
- Computational Graphs
- Reinforement Learning
- Arbiter

## Artifacts

NOT YET RELEASED TO CLOJARS
- fork or clone to try it out

If using Maven add the following repository definition to your pom.xml:

```
<repository>
  <id>clojars.org</id>
  <url>http://clojars.org/repo</url>
</repository>
```

## Latest release

With Leiningen:

```
n/a

```

With Maven:

n/a

```
<dependency>
  <groupId>_</groupId>
  <artifactId>_</artifactId>
  <version>_</version>
</dependency>
```

## Usage

### Things you need to know
- All functions for creating dl4j objects return code by default
  - All of these functions have an option to return the dl4j object
    - :as-code? = false
  - This because all builders require the code representation of dl4j objects
    - this requirement is not going to change
  - INDarray creation fns default to objects, this is for convenience
    - :as-code? is still respected

- API functions return code when all args are provided as code
- API functions return the value of calling the wrapped method when
  args are provided as a mixture of objects and code or just objects

- The tests are there to help clarify behavior, if you are unsure of how
  to use a fn, search the tests
  - for questions about spark, refer to the spark section bellow

### Example of obj/code duality

``` clojure
(ns my.ns
  (:require [dl4clj.nn.conf.builders.layers :as l]))

;; as code (the default)

(l/dense-layer-builder
 :activation-fn :relu
 :learning-rate 0.006
 :weight-init :xavier
 :layer-name "example layer"
 :n-in 10
 :n-out 1)

;; =>

(doto
 (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
 (.nOut 1)
 (.activation (dl4clj.constants/value-of {:activation-fn :relu}))
 (.weightInit (dl4clj.constants/value-of {:weight-init :xavier}))
 (.nIn 10)
 (.name "example layer")
 (.learningRate 0.006))

;; as an object

(l/dense-layer-builder
 :activation-fn :relu
 :learning-rate 0.006
 :weight-init :xavier
 :layer-name "example layer"
 :n-in 10
 :n-out 1
 :as-code? false)

;; =>

#object[org.deeplearning4j.nn.conf.layers.DenseLayer 0x69d7d160 "DenseLayer(super=FeedForwardLayer(super=Layer(layerName=example layer, activationFn=relu, weightInit=XAVIER, biasInit=NaN, dist=null, learningRate=0.006, biasLearningRate=NaN, learningRateSchedule=null, momentum=NaN, momentumSchedule=null, l1=NaN, l2=NaN, l1Bias=NaN, l2Bias=NaN, dropOut=NaN, updater=null, rho=NaN, epsilon=NaN, rmsDecay=NaN, adamMeanDecay=NaN, adamVarDecay=NaN, gradientNormalization=null, gradientNormalizationThreshold=NaN), nIn=10, nOut=1))"]

```

### General usage examples

#### Importing data

Loading data from a file (here its a csv)

``` clojure

(ns my.ns
 (:require [dl4clj.datasets.input-splits :as s]
           [dl4clj.datasets.record-readers :as rr]
           [dl4clj.datasets.api.record-readers :refer :all]
           [dl4clj.datasets.iterators :as ds-iter]
           [dl4clj.datasets.api.iterators :refer :all]
           [dl4clj.helpers :refer [data-from-iter]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file splits (convert the data to records)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def poker-path "resources/poker-hand-training.csv")
;; this is not a complete dataset, it is just here to sever as an example

(def file-split (s/new-filesplit :path poker-path))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; record readers, (read the records created by the file split)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def csv-rr (initialize-rr! :rr (rr/new-csv-record-reader :skip-n-lines 0 :delimiter ",")
                                 :input-split file-split))

;; lets look at some data
(println (next-record! :rr csv-rr :as-code? false))
;; => #object[java.util.ArrayList 0x2473e02d [1, 10, 1, 11, 1, 13, 1, 12, 1, 1, 9]]
;; this is our first line from the csv


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; record readers dataset iterators (turn our writables into a dataset)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def rr-ds-iter (ds-iter/new-record-reader-dataset-iterator
                 :record-reader csv-rr
                 :batch-size 1
                 :label-idx 10
                 :n-possible-labels 10))

;; we use our record reader created above
;; we want to see one example per dataset obj returned (:batch-size = 1)
;; we know our label is at the last index, so :label-idx = 10
;; there are 10 possible types of poker hands so :n-possible-labels = 10
;; you can also set :label-idx to -1 to use the last index no matter the size of the seq

(def other-rr-ds-iter (ds-iter/new-record-reader-dataset-iterator
                       :record-reader csv-rr
                       :batch-size 1
                       :label-idx -1
                       :n-possible-labels 10))

(str (next-example! :iter rr-ds-iter :as-code? false))
;; =>
;;===========INPUT===================
;;[1.00, 10.00, 1.00, 11.00, 1.00, 13.00, 1.00, 12.00, 1.00, 1.00]
;;=================OUTPUT==================
;;[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]


;; and to show that :label-idx = -1 gives us the same output

(= (next-example! :iter rr-ds-iter :as-code? false)
   (next-example! :iter other-rr-ds-iter :as-code? false)) ;; => true

```

#### INDArrays and Datasets from clojure data structures

``` clojure

(ns my.ns
  (:require [nd4clj.linalg.factory.nd4j :refer [vec->indarray matrix->indarray
                                                indarray-of-zeros indarray-of-ones
                                                indarray-of-rand vec-or-matrix->indarray]]
            [dl4clj.datasets.new-datasets :refer [new-ds]]
            [dl4clj.datasets.api.datasets :refer [as-list]]
            [dl4clj.datasets.iterators :refer [new-existing-dataset-iterator]]
            [dl4clj.datasets.api.iterators :refer :all]
            [dl4clj.datasets.pre-processors :as ds-pp]
            [dl4clj.datasets.api.pre-processors :refer :all]
            [dl4clj.core :as c]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; INDArray creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;TODO: consider defaulting to code

;; can create from a vector

(vec->indarray [1 2 3 4])
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x269df212 [1.00, 2.00, 3.00, 4.00]]

;; or from a matrix

(matrix->indarray [[1 2 3 4] [2 4 6 8]])
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x20aa7fe1
;; [[1.00, 2.00, 3.00, 4.00], [2.00, 4.00, 6.00, 8.00]]]


;; will fill in spareness with zeros

(matrix->indarray [[1 2 3 4] [2 4 6 8] [10 12]])
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x8b7796c
;;[[1.00, 2.00, 3.00, 4.00],
;; [2.00, 4.00, 6.00, 8.00],
;; [10.00, 12.00, 0.00, 0.00]]]

;; can create an indarray of all zeros with specified shape
;; defaults to :rows = 1 :columns = 1

(indarray-of-zeros :rows 3 :columns 2)
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x6f586a7e
;;[[0.00, 0.00],
;; [0.00, 0.00],
;; [0.00, 0.00]]]

(indarray-of-zeros) ;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0xe59ffec 0.00]

;; and if only one is supplied, will get a vector of specified length

(indarray-of-zeros :rows 2)
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x2899d974 [0.00, 0.00]]

(indarray-of-zeros :columns 2)
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0xa5b9782 [0.00, 0.00]]

;; same considerations/defaults for indarray-of-ones and indarray-of-rand

(indarray-of-ones :rows 2 :columns 3)
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x54f08662 [[1.00, 1.00, 1.00], [1.00, 1.00, 1.00]]]

(indarray-of-rand :rows 2 :columns 3)
;; all values are greater than 0 but less than 1
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x2f20293b [[0.85, 0.86, 0.13], [0.94, 0.04, 0.36]]]



;; vec-or-matrix->indarray is built into all functions which require INDArrays
;; so that you can use clojure data structures
;; but you still have the option of passing existing INDArrays

(def example-array (vec-or-matrix->indarray [1 2 3 4]))
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x5c44c71f [1.00, 2.00, 3.00, 4.00]]

(vec-or-matrix->indarray example-array)
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x607b03b0 [1.00, 2.00, 3.00, 4.00]]

(vec-or-matrix->indarray (indarray-of-rand :rows 2))
;; => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x49143b08 [0.76, 0.92]]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; data-set creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def ds-with-single-example (new-ds :input [1 2 3 4]
                                    :output [0.0 1.0 0.0]))

(as-list :ds ds-with-single-example :as-code? false)
;; =>
;; #object[java.util.ArrayList 0x5d703d12
;;[===========INPUT===================
;;[1.00, 2.00, 3.00, 4.00]
;;=================OUTPUT==================
;;[0.00, 1.00, 0.00]]]

(def ds-with-multiple-examples (new-ds
                                :input [[1 2 3 4] [2 4 6 8]]
                                :output [[0.0 1.0 0.0] [0.0 0.0 1.0]]))

(as-list :ds ds-with-multiple-examples :as-code? false)
;; =>
;;#object[java.util.ArrayList 0x29c7a9e2
;;[===========INPUT===================
;;[1.00, 2.00, 3.00, 4.00]
;;=================OUTPUT==================
;;[0.00, 1.00, 0.00],
;;===========INPUT===================
;;[2.00, 4.00, 6.00, 8.00]
;;=================OUTPUT==================
;;[0.00, 0.00, 1.00]]]

;; we can create a dataset iterator from the code which creates datasets
;; and set the labels for our outputs (optional)

(def ds-with-multiple-examples
  (new-ds
   :input [[1 2 3 4] [2 4 6 8]]
   :output [[0.0 1.0 0.0] [0.0 0.0 1.0]]))

;; iterator
(def training-rr-ds-iter
  (new-existing-dataset-iterator
   :dataset ds-with-multiple-examples
   :labels ["foo" "baz" "foobaz"]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; data-set normalization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; this gathers statistics on the dataset and normalizes the data
;; and applies the transformation to all dataset objects in the iterator
(def train-iter-normalized
  (c/normalize-iter! :iter training-rr-ds-iter
                     :normalizer (ds-pp/new-standardize-normalization-ds-preprocessor)
                     :as-code? false))

;; above returns the normalized iterator
;; to get fit normalizer

(def the-normalizer
  (get-pre-processor train-iter-normalized))

```

#### Model configuration

Creating a neural network configuration with singe and multiple layers

``` clojure
(ns my.ns
  (:require [dl4clj.nn.conf.builders.layers :as l]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.nn.conf.distributions :as dist]
            [dl4clj.nn.conf.input-pre-processor :as pp]
            [dl4clj.nn.conf.step-fns :as s-fn]))

;; nn/builder has 3 types of args
;; 1) args which set network configuration params
;; 2) args which set default values for layers
;; 3) args which set multi layer network configuration params

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; single layer nn configuration
;; here we are setting network configuration
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(nn/builder :optimization-algo :stochastic-gradient-descent
            :seed 123
            :iterations 1
            :minimize? true
            :use-drop-connect? false
            :lr-score-based-decay-rate 0.002
            :regularization? false
            :step-fn :default-step-fn
            :layers {:dense-layer {:activation-fn :relu
                                   :updater :adam
                                   :adam-mean-decay 0.2
                                   :adam-var-decay 0.1
                                   :learning-rate 0.006
                                   :weight-init :xavier
                                   :layer-name "single layer model example"
                                   :n-in 10
                                   :n-out 20}})

;; there are several options within a nn-conf map which can be configuration maps
;; or calls to fns
;; It doesn't matter which option you choose and you don't have to stay consistent
;; the list of params which can be passed as config maps or fn calls will
;; be enumerated at a later date

(nn/builder :optimization-algo :stochastic-gradient-descent
            :seed 123
            :iterations 1
            :minimize? true
            :use-drop-connect? false
            :lr-score-based-decay-rate 0.002
            :regularization? false
            :step-fn (s-fn/new-default-step-fn)
            :build? true
            ;; dont need to specify layer order, theres only one
            :layers (l/dense-layer-builder
                    :activation-fn :relu
                    :updater :adam
                    :adam-mean-decay 0.2
                    :adam-var-decay 0.1
                    :dist (dist/new-normal-distribution :mean 0 :std 1)
                    :learning-rate 0.006
                    :weight-init :xavier
                    :layer-name "single layer model example"
                    :n-in 10
                    :n-out 20))

;; these configurations are the same

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi-layer configuration
;; here we are also setting layer defaults
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; defaults will apply to layers which do not specify those value in their config

(nn/builder
 :optimization-algo :stochastic-gradient-descent
 :seed 123
 :iterations 1
 :minimize? true
 :use-drop-connect? false
 :lr-score-based-decay-rate 0.002
 :regularization? false
 :default-activation-fn :sigmoid
 :default-weight-init :uniform

 ;; we need to specify the layer order
 :layers {0 (l/activation-layer-builder
             :activation-fn :relu
             :updater :adam
             :adam-mean-decay 0.2
             :adam-var-decay 0.1
             :learning-rate 0.006
             :weight-init :xavier
             :layer-name "example first layer"
             :n-in 10
             :n-out 20)
          1 {:output-layer {:n-in 20
                            :n-out 2
                            :loss-fn :mse
                            :layer-name "example output layer"}}})

;; specifying multi-layer config params

(nn/builder
 ;; network args
 :optimization-algo :stochastic-gradient-descent
 :seed 123
 :iterations 1
 :minimize? true
 :use-drop-connect? false
 :lr-score-based-decay-rate 0.002
 :regularization? false

 ;; layer defaults
 :default-activation-fn :sigmoid
 :default-weight-init :uniform

 ;; the layers
 :layers {0 (l/activation-layer-builder
             :activation-fn :relu
             :updater :adam
             :adam-mean-decay 0.2
             :adam-var-decay 0.1
             :learning-rate 0.006
             :weight-init :xavier
             :layer-name "example first layer"
             :n-in 10
             :n-out 20)
          1 {:output-layer {:n-in 20
                            :n-out 2
                            :loss-fn :mse
                            :layer-name "example output layer"}}}
 ;; multi layer network args
 :backprop? true
 :backprop-type :standard
 :pretrain? false
 :input-pre-processors {0 (pp/new-zero-mean-pre-pre-processor)
                        1 {:unit-variance-processor {}}})

```

#### Configuration to Trained models

Multi Layer models
- an implementation of the dl4j [mnist classification example](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistTwoLayerExample.java)


``` clojure
(ns my.ns
  (:require [dl4clj.datasets.iterators :as iter]
            [dl4clj.datasets.input-splits :as split]
            [dl4clj.datasets.record-readers :as rr]
            [dl4clj.optimize.listeners :as listener]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.nn.multilayer.multi-layer-network :as mln]
            [dl4clj.nn.api.model :refer [init! set-listeners!]]
            [dl4clj.nn.api.multi-layer-network :refer [evaluate-classification]]
            [dl4clj.datasets.api.record-readers :refer [initialize-rr!]]
            [dl4clj.eval.api.eval :refer [get-stats get-accuracy]]
            [dl4clj.core :as c]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; nn-conf -> multi-layer-network
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def nn-conf
  (nn/builder
   ;; network args
   :optimization-algo :stochastic-gradient-descent
   :seed 123 :iterations 1 :regularization? true

   ;; setting layer defaults
   :default-activation-fn :relu :default-l2 7.5e-6
   :default-weight-init :xavier :default-learning-rate 0.0015
   :default-updater :nesterovs :default-momentum 0.98

   ;; setting layer configuration
   :layers {0 {:dense-layer
               {:layer-name "example first layer"
                :n-in 784 :n-out 500}}
            1 {:dense-layer
               {:layer-name "example second layer"
                :n-in 500 :n-out 100}}
            2 {:output-layer
               {:n-in 100 :n-out 10
                ;; layer specific params
                :loss-fn :negativeloglikelihood
                :activation-fn :softmax
                :layer-name "example output layer"}}}

   ;; multi layer args
   :backprop? true
   :pretrain? false))

(def multi-layer-network (c/model-from-conf nn-conf))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; local cpu training with dl4j pre-built iterators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; lets use the pre-built Mnist data set iterator

(def train-mnist-iter
  (iter/new-mnist-data-set-iterator
   :batch-size 64
   :train? true
   :seed 123))

(def test-mnist-iter
  (iter/new-mnist-data-set-iterator
   :batch-size 64
   :train? false
   :seed 123))

;; and lets set a listener so we can know how training is going

(def score-listener (listener/new-score-iteration-listener :print-every-n 5))

;; and attach it to our model

;; TODO: listeners are broken, look into log4j warnning
(def mln-with-listener (set-listeners! :model multi-layer-network
                                       :listeners [score-listener]))

(def trained-mln (mln/train-mln-with-ds-iter! :mln mln-with-listener
                                              :iter train-mnist-iter
                                              :n-epochs 15
                                              :as-code? false))

;; training happens because :as-code? = false
;; if it was true, we would still just have a data structure
;; we now have a trained model that has seen the training dataset 15 times
;; time to evaluate our model

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Create an evaluation object
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def eval-obj (evaluate-classification :mln trained-mln
                                       :iter test-mnist-iter))

;; always remember that these objects are stateful, dont use the same eval-obj
;; to eval two different networks
;; we trained the model on a training dataset.  We evaluate on a test set

(println (get-stats :evaler eval-obj))
;; this will print the stats to standard out for each feature/label pair

;;Examples labeled as 0 classified by model as 0: 968 times
;;Examples labeled as 0 classified by model as 1: 1 times
;;Examples labeled as 0 classified by model as 2: 1 times
;;Examples labeled as 0 classified by model as 3: 1 times
;;Examples labeled as 0 classified by model as 5: 1 times
;;Examples labeled as 0 classified by model as 6: 3 times
;;Examples labeled as 0 classified by model as 7: 1 times
;;Examples labeled as 0 classified by model as 8: 2 times
;;Examples labeled as 0 classified by model as 9: 2 times
;;Examples labeled as 1 classified by model as 1: 1126 times
;;Examples labeled as 1 classified by model as 2: 2 times
;;Examples labeled as 1 classified by model as 3: 1 times
;;Examples labeled as 1 classified by model as 5: 1 times
;;Examples labeled as 1 classified by model as 6: 2 times
;;Examples labeled as 1 classified by model as 7: 1 times
;;Examples labeled as 1 classified by model as 8: 2 times
;;Examples labeled as 2 classified by model as 0: 3 times
;;Examples labeled as 2 classified by model as 1: 2 times
;;Examples labeled as 2 classified by model as 2: 1006 times
;;Examples labeled as 2 classified by model as 3: 2 times
;;Examples labeled as 2 classified by model as 4: 3 times
;;Examples labeled as 2 classified by model as 6: 3 times
;;Examples labeled as 2 classified by model as 7: 7 times
;;Examples labeled as 2 classified by model as 8: 6 times
;;Examples labeled as 3 classified by model as 2: 4 times
;;Examples labeled as 3 classified by model as 3: 990 times
;;Examples labeled as 3 classified by model as 5: 3 times
;;Examples labeled as 3 classified by model as 7: 3 times
;;Examples labeled as 3 classified by model as 8: 3 times
;;Examples labeled as 3 classified by model as 9: 7 times
;;Examples labeled as 4 classified by model as 2: 2 times
;;Examples labeled as 4 classified by model as 3: 1 times
;;Examples labeled as 4 classified by model as 4: 967 times
;;Examples labeled as 4 classified by model as 6: 4 times
;;Examples labeled as 4 classified by model as 7: 1 times
;;Examples labeled as 4 classified by model as 9: 7 times
;;Examples labeled as 5 classified by model as 0: 2 times
;;Examples labeled as 5 classified by model as 3: 6 times
;;Examples labeled as 5 classified by model as 4: 1 times
;;Examples labeled as 5 classified by model as 5: 874 times
;;Examples labeled as 5 classified by model as 6: 3 times
;;Examples labeled as 5 classified by model as 7: 1 times
;;Examples labeled as 5 classified by model as 8: 3 times
;;Examples labeled as 5 classified by model as 9: 2 times
;;Examples labeled as 6 classified by model as 0: 4 times
;;Examples labeled as 6 classified by model as 1: 3 times
;;Examples labeled as 6 classified by model as 3: 2 times
;;Examples labeled as 6 classified by model as 4: 4 times
;;Examples labeled as 6 classified by model as 5: 4 times
;;Examples labeled as 6 classified by model as 6: 939 times
;;Examples labeled as 6 classified by model as 7: 1 times
;;Examples labeled as 6 classified by model as 8: 1 times
;;Examples labeled as 7 classified by model as 1: 7 times
;;Examples labeled as 7 classified by model as 2: 4 times
;;Examples labeled as 7 classified by model as 3: 3 times
;;Examples labeled as 7 classified by model as 7: 1005 times
;;Examples labeled as 7 classified by model as 8: 2 times
;;Examples labeled as 7 classified by model as 9: 7 times
;;Examples labeled as 8 classified by model as 0: 3 times
;;Examples labeled as 8 classified by model as 2: 3 times
;;Examples labeled as 8 classified by model as 3: 2 times
;;Examples labeled as 8 classified by model as 4: 4 times
;;Examples labeled as 8 classified by model as 5: 3 times
;;Examples labeled as 8 classified by model as 6: 2 times
;;Examples labeled as 8 classified by model as 7: 4 times
;;Examples labeled as 8 classified by model as 8: 947 times
;;Examples labeled as 8 classified by model as 9: 6 times
;;Examples labeled as 9 classified by model as 0: 2 times
;;Examples labeled as 9 classified by model as 1: 2 times
;;Examples labeled as 9 classified by model as 3: 4 times
;;Examples labeled as 9 classified by model as 4: 8 times
;;Examples labeled as 9 classified by model as 6: 1 times
;;Examples labeled as 9 classified by model as 7: 4 times
;;Examples labeled as 9 classified by model as 8: 2 times
;;Examples labeled as 9 classified by model as 9: 986 times

;;==========================Scores========================================
;; Accuracy:        0.9808
;; Precision:       0.9808
;; Recall:          0.9807
;; F1 Score:        0.9807
;;========================================================================

;; can get the stats that are printed via fns in the evaluation namespace
;; after running eval-model-whole-ds

(get-accuracy :evaler evaler-with-stats) ;; => 0.9808


```

#### Model Tuning

Early Stopping (controlling training)
- it is recommened you start here when designing models

- using dl4clj.core

``` clojure

(ns my.ns
  (:require [dl4clj.earlystopping.termination-conditions :refer :all]
            [dl4clj.earlystopping.model-saver :refer [new-in-memory-saver]]
            [dl4clj.nn.api.multi-layer-network :refer [evaluate-classification]]
            [dl4clj.eval.api.eval :refer [get-stats]]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.datasets.iterators :as iter]
            [dl4clj.core :as c]))

(def nn-conf
  (nn/builder
   ;; network args
   :optimization-algo :stochastic-gradient-descent
   :seed 123
   :iterations 1
   :regularization? true

   ;; setting layer defaults
   :default-activation-fn :relu
   :default-l2 7.5e-6
   :default-weight-init :xavier
   :default-learning-rate 0.0015
   :default-updater :nesterovs
   :default-momentum 0.98

   ;; setting layer configuration
   :layers {0 {:dense-layer
               {:layer-name "example first layer"
                :n-in 784 :n-out 500}}
            1 {:dense-layer
               {:layer-name "example second layer"
                :n-in 500 :n-out 100}}
            2 {:output-layer
               {:n-in 100 :n-out 10
                ;; layer specific params
                :loss-fn :negativeloglikelihood
                :activation-fn :softmax
                :layer-name "example output layer"}}}

   ;; multi layer args
   :backprop? true
   :pretrain? false))

(def train-iter
  (iter/new-mnist-data-set-iterator
   :batch-size 64
   :train? true
   :seed 123))

(def test-iter
  (iter/new-mnist-data-set-iterator
   :batch-size 64
   :train? false
   :seed 123))

(def invalid-score-condition (new-invalid-score-iteration-termination-condition))

(def max-score-condition (new-max-score-iteration-termination-condition
                          :max-score 20.0))

(def max-time-condition (new-max-time-iteration-termination-condition
                         :max-time-val 10
                         :max-time-unit :minutes))

(def score-doesnt-improve-condition (new-score-improvement-epoch-termination-condition
                                     :max-n-epoch-no-improve 5))

(def target-score-condition (new-best-score-epoch-termination-condition
                             :best-expected-score 0.009))

(def max-number-epochs-condition (new-max-epochs-termination-condition :max-n 20))

(def in-mem-saver (new-in-memory-saver))

(def trained-mln
;; defaults to returning the model
  (c/train-with-early-stopping
   :nn-conf nn-conf
   :training-iter train-mnist-iter
   :testing-iter test-mnist-iter
   :eval-every-n-epochs 1
   :iteration-termination-conditions [invalid-score-condition
                                      max-score-condition
                                      max-time-condition]
   :epoch-termination-conditions [score-doesnt-improve-condition
                                  target-score-condition
                                  max-number-epochs-condition]
   :save-last-model? true
   :model-saver in-mem-saver
   :as-code? false))

(def model-evaler
  (evaluate-classification :mln trained-mln :iter test-mnist-iter))

(println (get-stats :evaler model-evaler))

```

- explicit, step by step way of doing this

``` clojure
(ns my.ns
  (:require [dl4clj.earlystopping.early-stopping-config :refer [new-early-stopping-config]]
            [dl4clj.earlystopping.termination-conditions :refer :all]
            [dl4clj.earlystopping.model-saver :refer [new-in-memory-saver new-local-file-model-saver]]
            [dl4clj.earlystopping.score-calc :refer [new-ds-loss-calculator]]
            [dl4clj.earlystopping.early-stopping-trainer :refer [new-early-stopping-trainer]]
            [dl4clj.earlystopping.api.early-stopping-trainer :refer [fit-trainer!]]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.nn.multilayer.multi-layer-network :as mln]
            [dl4clj.utils :refer [load-model!]]
            [dl4clj.datasets.iterators :as iter]
            [dl4clj.core :as c]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; start with our network config
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def nn-conf
  (nn/builder
   ;; network args
   :optimization-algo :stochastic-gradient-descent
   :seed 123 :iterations 1 :regularization? true
   ;; setting layer defaults
   :default-activation-fn :relu :default-l2 7.5e-6
   :default-weight-init :xavier :default-learning-rate 0.0015
   :default-updater :nesterovs :default-momentum 0.98
   ;; setting layer configuration
   :layers {0 {:dense-layer
               {:layer-name "example first layer"
                :n-in 784 :n-out 500}}
            1 {:dense-layer
               {:layer-name "example second layer"
                :n-in 500 :n-out 100}}
            2 {:output-layer
               {:n-in 100 :n-out 10
                ;; layer specific params
                :loss-fn :negativeloglikelihood
                :activation-fn :softmax
                :layer-name "example output layer"}}}
   ;; multi layer args
   :backprop? true
   :pretrain? false))

(def mln (c/model-from-conf nn-conf))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; the training/testing data
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def train-iter
  (iter/new-mnist-data-set-iterator
   :batch-size 64
   :train? true
   :seed 123))

(def test-iter
  (iter/new-mnist-data-set-iterator
   :batch-size 64
   :train? false
   :seed 123))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; we are going to need termination conditions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; these allow us to control when we exit training

;; this can be based off of iterations or epochs

;; iteration termination conditions

(def invalid-score-condition (new-invalid-score-iteration-termination-condition))

(def max-score-condition (new-max-score-iteration-termination-condition
                          :max-score 20.0))

(def max-time-condition (new-max-time-iteration-termination-condition
                         :max-time-val 10
                         :max-time-unit :minutes))

;; epoch termination conditions

(def score-doesnt-improve-condition (new-score-improvement-epoch-termination-condition
                                     :max-n-epoch-no-improve 5))

(def target-score-condition (new-best-score-epoch-termination-condition :best-expected-score 0.009))

(def max-number-epochs-condition (new-max-epochs-termination-condition :max-n 20))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; we also need a way to save our model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; can be in memory or to a local directory

(def in-mem-saver (new-in-memory-saver))

(def local-file-saver (new-local-file-model-saver :directory "resources/tmp/readme/"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; set up your score calculator
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def score-calcer (new-ds-loss-calculator :iter test-iter
                                          :average? true))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; create an early stopping configuration
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; termination conditions
;; a way to save our model
;; a way to calculate the score of our model on the dataset

(def early-stopping-conf
  (new-early-stopping-config
   :epoch-termination-conditions [score-doesnt-improve-condition
                                  target-score-condition
                                  max-number-epochs-condition]
   :iteration-termination-conditions [invalid-score-condition
                                      max-score-condition
                                      max-time-condition]
   :eval-every-n-epochs 5
   :model-saver local-file-saver
   :save-last-model? true
   :score-calculator score-calcer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; create an early stopping trainer from our data, model and early stopping conf
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def es-trainer (new-early-stopping-trainer :early-stopping-conf early-stopping-conf
                                            :mln mln
                                            :iter train-iter))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fit and use our early stopping trainer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def es-trainer-fitted (fit-trainer! es-trainer :as-code? false))

;; when the trainer terminates, you will see something like this
;;[nREPL-worker-24] BaseEarlyStoppingTrainer INFO  Completed training epoch 14
;;[nREPL-worker-24] BaseEarlyStoppingTrainer INFO  New best model: score = 0.005225599372851298,
;;                                                   epoch = 14 (previous: score = 0.018243224899038346, epoch = 7)
;;[nREPL-worker-24] BaseEarlyStoppingTrainer INFO Hit epoch termination condition at epoch 14.
;;                                           Details: BestScoreEpochTerminationCondition(0.009)

;; and if we look at the es-trainer-fitted object we see

;;#object[org.deeplearning4j.earlystopping.EarlyStoppingResult 0x5ab74f27 EarlyStoppingResult
;;(terminationReason=EpochTerminationCondition,details=BestScoreEpochTerminationCondition(0.009),
;; bestModelEpoch=14,bestModelScore=0.005225599372851298,totalEpochs=15)]

;; and our model has been saved to /resources/tmp/readme/bestModel.bin
;; there we have our model config, model params and our updater state

;; we can then load this model to use it or continue refining it

(def loaded-model (load-model! :path "resources/tmp/readme/bestModel.bin"
                               :load-updater? true))

```

Transfer Learning (freezing layers)

``` clojure

;; TODO: need to write up examples

```


### Spark Training
dl4j [Spark](https://deeplearning4j.org/spark) usage

How it is done in dl4clj

- Uses dl4clj.core
  - This example uses a fn which takes care of most steps for you
    - allows you to pass args as code bc the fn accounts for
      the multiple spark contexts issue encountered when everything
      is just a data structure

``` clojure

(ns my.ns
  (:require [dl4clj.nn.conf.builders.layers :as l]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.datasets.iterators :refer [new-iris-data-set-iterator]]
            [dl4clj.eval.api.eval :refer [get-stats]]
            [dl4clj.spark.masters.param-avg :as master]
            [dl4clj.spark.data.java-rdd :refer [new-java-spark-context
                                                java-rdd-from-iter]]
            [dl4clj.spark.api.dl4j-multi-layer :refer [eval-classification-spark-mln
                                                       get-spark-context]]
            [dl4clj.core :as c]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 1, create your model config
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def mln-conf
  (nn/builder
   :optimization-algo :stochastic-gradient-descent
   :default-learning-rate 0.006
   :layers {0 (l/dense-layer-builder :n-in 4 :n-out 2 :activation-fn :relu)
            1 {:output-layer
               {:loss-fn :negativeloglikelihood
                :n-in 2 :n-out 3
                :activation-fn :soft-max
                :weight-init :xavier}}}
   :backprop? true
   :backprop-type :standard))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 2, training master
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def training-master
  (master/new-parameter-averaging-training-master
   :build? true
   :rdd-n-examples 10
   :n-workers 4
   :averaging-freq 10
   :batch-size-per-worker 2
   :export-dir "resources/spark/master/"
   :rdd-training-approach :direct
   :repartition-data :always
   :repartition-strategy :balanced
   :seed 1234
   :save-updater? true
   :storage-level :none))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 3, spark context
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def your-spark-context
  (new-java-spark-context :app-name "example app"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 4, training data
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def iris-iter
  (new-iris-data-set-iterator
   :batch-size 1
   :n-examples 5))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 5, spark mln
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def fitted-spark-mln
  (c/train-with-spark :spark-context your-spark-context
                      :mln-conf mln-conf
                      :training-master training-master
                      :iter iris-iter
                      :n-epochs 1
                      :as-code? false))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 5, use spark context from spark-mln to create rdd
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; TODO: eliminate this step

(def our-rdd
  (let [sc (get-spark-context fitted-spark-mln :as-code? false)]
    (java-rdd-from-iter :spark-context sc
                        :iter iris-iter)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 6, evaluation model and print stats (poor performance of model expected)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def eval-obj
  (eval-classification-spark-mln
   :spark-mln fitted-spark-mln
   :rdd our-rdd))

(println (get-stats :evaler eval-obj))


```

- this example demonstrates the dl4j [workflow](https://deeplearning4j.org/spark#Overview)
  - NOTE: unlike the previous example, this one requires dl4j objects to be used
    - this is becaues spark only wants you to have one spark context at a time

``` clojure
(ns my.ns
  (:require [dl4clj.nn.conf.builders.layers :as l]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.datasets.iterators :refer [new-iris-data-set-iterator]]
            [dl4clj.eval.api.eval :refer [get-stats]]
            [dl4clj.spark.masters.param-avg :as master]
            [dl4clj.spark.data.java-rdd :refer [new-java-spark-context java-rdd-from-iter]]
            [dl4clj.spark.dl4j-multi-layer :as spark-mln]
            [dl4clj.spark.api.dl4j-multi-layer :refer [fit-spark-mln!
                                                       eval-classification-spark-mln]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 1, create your model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def mln-conf
  (nn/builder
   :optimization-algo :stochastic-gradient-descent
   :default-learning-rate 0.006
   :layers {0 (l/dense-layer-builder :n-in 4 :n-out 2 :activation-fn :relu)
            1 {:output-layer
               {:loss-fn :negativeloglikelihood
                :n-in 2 :n-out 3
                :activation-fn :soft-max
                :weight-init :xavier}}}
   :backprop? true
   :as-code? false
   :backprop-type :standard))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 2, create a training master
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; not all options specified, but most are

(def training-master
  (master/new-parameter-averaging-training-master
   :build? true
   :rdd-n-examples 10
   :n-workers 4
   :averaging-freq 10
   :batch-size-per-worker 2
   :export-dir "resources/spark/master/"
   :rdd-training-approach :direct
   :repartition-data :always
   :repartition-strategy :balanced
   :seed 1234
   :as-code? false
   :save-updater? true
   :storage-level :none))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 3, create a Spark Multi Layer Network
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def your-spark-context
  (new-java-spark-context :app-name "example app" :as-code? false))

;; new-java-spark-context will turn an existing spark-configuration into a java spark context
;; or create a new java spark context with master set to "local[*]" and the app name
;; set to :app-name


(def spark-mln
  (spark-mln/new-spark-multi-layer-network
   :spark-context your-spark-context
   :mln mln-conf
   :training-master training-master
   :as-code? false))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 4, load your data
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; one way is via a dataset-iterator
;; can make one directly from a dataset (iterator data-set)
;; see: nd4clj.linalg.dataset.api.data-set and nd4clj.linalg.dataset.data-set
;; we are going to use a pre-built one

(def iris-iter
  (new-iris-data-set-iterator
   :batch-size 1
   :n-examples 5
   :as-code? false))

;; now lets convert the data into a javaRDD

(def our-rdd
  (java-rdd-from-iter :spark-context your-spark-context
                      :iter iris-iter))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 5, fit and evaluate the model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def fitted-spark-mln
  (fit-spark-mln!
   :spark-mln spark-mln
   :rdd our-rdd
   :n-epochs 1))
;; this fn also has the option to supply :path-to-data instead of :rdd
;; that path should point to a directory containing a number of dataset objects

(def eval-obj
  (eval-classification-spark-mln
   :spark-mln fitted-spark-mln
   :rdd our-rdd))
;; we would want to have different testing and training rdd's but here we are using
;; the data we trained on

;; lets get the stats for how our model performed

(println (get-stats :evaler eval-obj))

```
## Terminology

Coming soon

## Packages to come back to:

Implement ComputationGraphs and the classes which use them
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ComputationGraphConfiguration.GraphBuilder.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ComputationGraphConfiguration.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/graph/package-frame.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/graph/rnn/package-frame.html>

NLP
- <https://deeplearning4j.org/doc/org/deeplearning4j/bagofwords/vectorizer/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/datasets/vectorizer/Vectorizer.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/iterator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/iterator/provider/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/WeightLookupTable.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/inmemory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/learning/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/learning/impl/elements/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/learning/impl/sequence/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/loader/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/reader/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/reader/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/embeddings/wordvectors/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/glove/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/glove/count/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/node2vec/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/paragraphvectors/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/enums/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/interfaces/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/iterators/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/listeners/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/sequence/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/serialization/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/transformers/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/transformers/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/transformers/impl/iterables/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/iterator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/wordstore/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/wordstore/inmemory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/annotator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/corpora/sentiwordnet/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/corpora/treeparser/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/corpora/treeparser/transformer/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/documentiterator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/documentiterator/interoperability/DocumentIteratorConverter.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/inputsanitation/InputHomogenization.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/invertedindex/InvertedIndex.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/labels/LabelsProvider.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/movingwindow/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/interoperability/SentenceIteratorConverter.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/labelaware/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/stopwords/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/preprocessor/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizer/tokenprepreprocessor/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/tokenization/tokenizerfactory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/text/uima/UimaResource.html>

Parallelism
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/factory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/main/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/parameterserver/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/trainer/package-summary.html>

TSNE
- <https://deeplearning4j.org/doc/org/deeplearning4j/plot/package-summary.html>

UI
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/activation/PathUpdate.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/api/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/chart/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/chart/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/component/ComponentDiv.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/component/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/decorator/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/decorator/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/table/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/table/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/text/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/components/text/style/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/flow/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/flow/beans/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/flow/data/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/i18n/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/convolutional/ConvolutionalListenerModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/defaultModule/DefaultModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/flow/FlowListenerModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/histogram/HistogramModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/remote/RemoteReceiverModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/train/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/module/tsne/TsneModule.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/nearestneighbors/word2vec/NearestNeighborsQuery.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/play/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/play/misc/FunctionUtil.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/play/staticroutes/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/providers/ObjectMapperProvider.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/renders/PathUpdate.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/standalone/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/api/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/impl/java/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/stats/sbe/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/storage/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/storage/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/storage/mapdb/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/storage/sqlite/J7FileStatsStorage.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/weights/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/ui/weights/beans/CompactModelAndGradient.html>

## License

Copyright © 2016 Engagor

Distributed under the BSD Clause-2 License as distributed in the file LICENSE at the root of this repository.

![](https://thumbs.gfycat.com/IncompleteAncientJanenschia-size_restricted.gif)

*Drops the mic*
