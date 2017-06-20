# dl4clj

Port of [deeplearning4j](https://github.com/deeplearning4j/) to clojure

## Features

Not all of these are fully tested and are most likely going to undergo breaking changes
- the tested features are stable for now and should stay that way.
- even in the tested name spaces, there remain some functions which are either
  - broken
  - misunderstood
  - not tested

- Clustering
- Datasets (tested)
- Early Stopping (tested)
- Eval/Evaluation (tested)
- Neural Networks DSL (tested)
- Optimize (tested)
- Spark training/hosting

Support for Computational Graphs will come in a future release

## Artifacts

dl4clj artifacts are released to Clojars. (original authors work)

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
[engagor/clj-vw "0.0.1"]

```

With Maven:

```
<dependency>
  <groupId>engagor</groupId>
  <artifactId>dl4clj</artifactId>
  <version>0.0.1</version>
</dependency>
```

## Usage

### Layers

Creating distributions to sample layer weights from

``` clojure

(ns my.ns
  (:require [dl4clj.nn.conf.distribution.distribution :as dist]))

(dist/new-normal-distribution :mean 0 :std 1)

(dist/new-binomial-distribution :number-of-trials 5 :probability-of-success 0.50)

```

Creating Layers

``` clojure
(ns my.ns
  (:require [dl4clj.nn.conf.builders.builders :as l]))

(l/activation-layer-builder
 :activation-fn :relu :updater :adam
 :adam-mean-decay 0.2 :adam-var-decay 0.1
 :dist {:normal {:mean 0 :std 1.0}}
 :learning-rate 0.006 :weight-init :xavier
 :layer-name "example layer" :n-in 10 :n-out 1)

(l/activation-layer-builder
 :activation-fn :relu :updater :adam
 :adam-mean-decay 0.2 :adam-var-decay 0.1
 :dist (dist/new-normal-distribution :mean 0 :std 1)
 :learning-rate 0.006 :weight-init :xavier
 :layer-name "example layer" :n-in 10 :n-out 1)

;;these layer configurations are the same
```

There is also support for Variational Autoencoders

``` clojure
(ns my.ns
  (:require [dl4clj.nn.conf.builders.builders :as l]
            [dl4clj.nn.conf.variational.dist-builders :as v-dist]))

(l/variational-autoencoder-builder
 :decoder-layer-sizes [2 2]
 :encoder-layer-sizes [2 2]
 :reconstruction-distribution {:bernoulli {:activation-fn :sigmoid}}
 :pzx-activation-function :identity)

(l/variational-autoencoder-builder
 :decoder-layer-sizes [2 2]
 :encoder-layer-sizes [2 2]
 :reconstruction-distribution (v-dist/new-bernoulli-reconstruction-distribution
                               :activation-fn :sigmoid)
 :pzx-activation-function :identity)

;;these configurations are the same
```

There are a lot of utilities for working with Convolutional and Recurrent layers
- see: dl4clj.nn.conf.layers.input-type-util

And working with any type of layer
- see: dl4clj.nn.conf.layers.shared-fns
  - need to revisit this ns. make sure all the language is clear and no overlapping
    fns with the model interface ns

There is also configuration validation
- see: dl4clj.nn.conf.layers.layer-testing.layer-validation

### Model configuration

Creating input pre-processors

``` clojure

(ns my.ns
  (:require [dl4clj.nn.conf.input-pre-processor :as pp]))

;; single pre-processor

(pp/new-cnn-to-feed-forward-pre-processor
 :input-height 2 :input-width 3 :num-channels 4)

;; single pre-processor made of multiple pre-processors
;; this fn also supports heterogeneous args

(pp/new-composable-input-pre-processor
 :pre-processors [(new-zero-mean-pre-pre-processor)
                  (new-binominal-sampling-pre-processor)
                  {:cnn-to-feed-forward-pre-processor
                   {:input-height 2 :input-width 3 :num-channels 4}}])

```

Adding the layers to a neural network configuration

``` clojure
(ns my.ns
  (:require [dl4clj.nn.conf.builders.builders :as l]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn-conf]
            [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]
            [dl4clj.nn.conf.distribution.distribution :as dist]
            [dl4clj.nn.conf.step-fns :as s-fn]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; single layer nn-conf
;; build? should be set to true
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(nn-conf/nn-conf-builder :optimization-algo :stochastic-gradient-descent
                         :seed 123 :iterations 1 :minimize? true
                         :use-drop-connect? false :lr-score-based-decay-rate 0.002
                         :regularization? false
                         :step-fn (s-fn/new-default-step-fn)
                         :build? true
                         :layer (l/dense-layer-builder
                                 :activation-fn :relu :updater :adam
                                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                                 :dist {:normal {:mean 0 :std 1.0}}
                                 :learning-rate 0.006 :weight-init :xavier
                                 :layer-name "single layer model example"
                                 :n-in 10 :n-out 20))

;; distributions can be created using the fns in dl4clj.nn.conf.distribution.distribution
;; or by passing a configuration map as shown above

(nn-conf/nn-conf-builder :optimization-algo :stochastic-gradient-descent
                         :seed 123 :iterations 1 :minimize? true
                         :use-drop-connect? false :lr-score-based-decay-rate 0.002
                         :regularization? false
                         :step-fn :default-step-fn
                         :build? true
                         :layer (l/dense-layer-builder
                                 :activation-fn :relu :updater :adam
                                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                                 :dist (dist/new-normal-distribution :mean 0 :std 1)
                                 :learning-rate 0.006 :weight-init :xavier
                                 :layer-name "single layer model example"
                                 :n-in 10 :n-out 20))

;; these configurations are the same

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; model with multiple layers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; when we dont want to specify multi-layer-config-builder options
;; build? set to true

(nn-conf/nn-conf-builder :optimization-algo :stochastic-gradient-descent
                         :seed 123 :iterations 1 :minimize? true
                         :use-drop-connect? false :lr-score-based-decay-rate 0.002
                         :regularization? false
                         :default-activation-fn :sigmoid :default-weight-init :uniform
                         ;; ^ these defaults will be applied to "final layer" but will
                         ;; not overwrite "first of two layers"
                         :build? true
                         ;; :layers accepts heterogeneous config maps
                         :layers {0 (l/activation-layer-builder
                                     :activation-fn :relu :updater :adam
                                     :adam-mean-decay 0.2 :adam-var-decay 0.1
                                     :dist (new-normal-distribution :mean 0 :std 1)
                                     :learning-rate 0.006 :weight-init :xavier
                                     :layer-name "first of two layers" :n-in 10 :n-out 20)
                                  1 {:output-layer {:n-in 20 :n-out 2 :loss-fn :mse
                                                    :layer-name "final layer"}}})


;; when we want to specify multi-layer-config-builder options
;; can leave out :build? as it defaults to false

(def l-builder (nn-conf/nn-conf-builder
                :optimization-algo :stochastic-gradient-descent
                :seed 123 :iterations 1 :minimize? true
                :use-drop-connect? false :lr-score-based-decay-rate 0.002
                :regularization? false
                :default-activation-fn :sigmoid :default-weight-init :uniform
                :layers {0 (l/activation-layer-builder
                            :activation-fn :relu :updater :adam
                            :adam-mean-decay 0.2 :adam-var-decay 0.1
                            :dist {:normal {:mean 0 :std 1.0}}
                            :learning-rate 0.006 :weight-init :xavier
                            :layer-name "example first layer" :n-in 10 :n-out 20)
                         1 {:output-layer {:n-in 20 :n-out 2 :loss-fn :mse
                                           :layer-name "example output layer"}}}))

;; here we can add in pre-processors
;; pass in a map of either pre-processor fn calls or configuration maps
;; can be heterogeneous just like any other place where you can pass fns or config maps

(mlb/multi-layer-config-builder
 :list-builder l-builder
 :backprop? true :backprop-type :standard
 :pretrain? false
 :input-pre-processors {0 (new-zero-mean-pre-pre-processor)
                        1 {:unit-variance-processor {}})

;; we can also create multi-layer configurations using many single layer configurations

(def first-layer-conf
  (nn-conf/nn-conf-builder
   :optimization-algo :stochastic-gradient-descent
   :seed 123 :iterations 1 :minimize? true
   :use-drop-connect? false :lr-score-based-decay-rate 0.002
   :regularization? false
   :build? true
   :layer {:dense-layer {:activation-fn :relu :updater :adam
                         :adam-mean-decay 0.2 :adam-var-decay 0.1
                         :dist {:normal {:mean 0 :std 1.0}}
                         :learning-rate 0.006 :weight-init :xavier
                         :layer-name "first layer"
                         :n-in 10 :n-out 20}}))

(def second-layer-conf
  (nn-conf/nn-conf-builder
   :optimization-algo :stochastic-gradient-descent
   :seed 123 :iterations 1 :minimize? true
   :use-drop-connect? false :lr-score-based-decay-rate 0.002
   :regularization? false
   :build? true
   :layer {:output-layer {:n-in 20 :n-out 2 :loss-fn :mse
                          :layer-name "second layer"}}))

(def multi-from-multiple-single
 (multi-layer-config-builder :nn-confs [first-layer-conf second-layer-conf]))

```

### Importing data

Loading data from a file (here its a csv)

``` clojure

(my.ns (:require [datavec.api.split :as s]
                 [datavec.api.records.readers :as rr]
                 [datavec.api.records.interface :refer :all]
                 [dl4clj.datasets.datavec :as ds-iter]
                 [nd4clj.linalg.api.ds-iter :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; file splits (convert the data to records)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def poker-path "resources/poker/poker-hand-training.csv")

(def file-split (s/new-filesplit :path poker-path))

;; we can't look at the data until we create a record reader

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; record readers, (read the records created by the file split)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def csv-rr (initialize-rr! :rr (rr/new-csv-record-reader :skip-num-lines 0 :delimiter ",")
                                 :input-split file-split))

;; lets look at some data
(println (next-record! csv-rr))
;; => #object[java.util.ArrayList 0x2473e02d [1, 10, 1, 11, 1, 13, 1, 12, 1, 1, 9]]
;; this is our first line from the csv

;; next-record! moves the record readers interal cursor, so we should now reset the record reader

(reset-rr! csv-rr)
;; this will return the reset record reader, so this fn can be at the start of a fn chain

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

;; lets look at some data

(str (next-example! rr-ds-iter))
;; =>
;;===========INPUT===================
;;[1.00, 10.00, 1.00, 11.00, 1.00, 13.00, 1.00, 12.00, 1.00, 1.00]
;;=================OUTPUT==================
;;[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]

;; dont forget to reset the iterator as next example changes the cursor value
(reset-iter! rr-ds-iter)

;; and to show that :label-idx = -1 gives us the same output

(= (next-example! (reset-iter! rr-ds-iter))
   (next-example! (reset-iter! other-rr-ds-iter)))

;; => true

;; we now have our csv data in a format that can be fed to a neural network

;; if we want all the data out of our record-reader-dataset-iterator as a lazy seq

(def lazy-seq-data (data-from-iter rr-ds-iter))

(realized? lazy-seq-data) ;; => false

(first lazy-seq-data) ;; =>

;;===========INPUT===================
;;[1.00, 10.00, 1.00, 11.00, 1.00, 13.00, 1.00, 12.00, 1.00, 1.00]
;;=================OUTPUT==================
;;[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]

;; getting the data out of a rr-ds-iter ourself becomes important
;; when making javaRDDs for spark training but is not necessary
;; for local machine training

```

Creating datasets from INDArrays (and creating INDArrays)

``` clojure

(my.ns
 (:require [nd4clj.linalg.factory.nd4j :refer [vec->indarray matrix->indarray
                                               indarray-of-zeros indarray-of-ones
                                               indarray-of-rand]]
           [nd4clj.linalg.dataset.data-set :refer [data-set]]
           [nd4clj.linalg.dataset.api.data-set :refer [as-list]]
           [dl4clj.datasets.iterator.iterators :refer[new-existing-dataset-iterator]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; INDArray creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

(indarray-of-zeros) => #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0xe59ffec 0.00]

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; data-set creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def ds-with-single-example (data-set :input (vec->indarray [1 2 3 4])
                                      :output (vec->indarray [0.0 1.0 0.0])))
(as-list ds-with-single-example)
;; =>
;; #object[java.util.ArrayList 0x5d703d12
;;[===========INPUT===================
;;[1.00, 2.00, 3.00, 4.00]
;;=================OUTPUT==================
;;[0.00, 1.00, 0.00]]]

(def ds-with-multiple-examples (data-set
                                :input (matrix->indarray [[1 2 3 4] [2 4 6 8]])
                                :output (matrix->indarray [[0.0 1.0 0.0] [0.0 0.0 1.0]])))

(as-list ds-with-multiple-examples)
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

;; we can create a dataset iterator directly from a dataset
;; and set the labels for our outputs (optional)
(new-existing-dataset-iterator :dataset ds-with-multiple-examples :labels ["foo" "baz" "foobaz"])

```

### Congiuration to Initialized models

Layers as models

``` clojure

dl4clj.nn.layers.layer-creation


```

Multi Layer models

``` clojure

dl4clj.nn.multilayer.multi-layer-network


```

Evaluation of Models

``` clojure




```
### Model Tuning

Transfer Learning (freezing layers)

``` clojure




```

Early Stopping (controlling training)

``` clojure




```

### Spark Training
For a walk through on how to use Spark wtih dl4j, see: https://deeplearning4j.org/spark

How it is done in dl4clj
- same workflow as https://deeplearning4j.org/spark#Overview
  - NOTE: need to verify the spark hosting of trained models

``` clojure
(ns my.ns
  (:require [dl4clj.nn.conf.builders.builders :as l]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn-conf]
            [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]
            [dl4clj.nn.multilayer.multi-layer-network :as mln]
            [dl4clj.spark.masters.param-avg :as master]
            [dl4clj.spark.dl4j-multi-layer :as spark-mln]
            [dl4clj.datasets.iterator.impl.default-datasets :refer [new-iris-data-set-iterator]]
            [dl4clj.spark.data.java-rdd :refer [new-java-spark-context java-rdd-from-iter]]
            [dl4clj.spark.dl4j-layer :refer [new-spark-dl4j-layer fit-spark-layer-with-ds!]]
            [dl4clj.eval.evaluation :refer [get-stats]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 1, create your model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def mln-conf
  (nn-conf/nn-conf-builder
   :optimization-algo :stochastic-gradient-descent
   :learning-rate 0.006
   :build? false
   :layers {0 (l/dense-layer-builder :n-in 10 :n-out 2 :activation-fn :relu)
            1 {:output-layer
               {:loss-fn :negativeloglikelihood
                :n-in 2 :n-out 1
                :activation-fn :soft-max
                :weight-init :xavier}}}))

(def multi-layer-model
  (mln/new-multi-layer-network :conf
   (mlb/multi-layer-config-builder
    :list-builder mln-conf
    :backprop? true
    :backprop-type :standard)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 2, create a training master
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; not all options specified, but most are

(def training-master
 (master/new-parameter-averaging-training-master
 :build? true :rdd-n-examples 10 :n-workers 4 :averaging-freq 10
 :batch-size-per-worker 2 :export-dir "resources/spark/master/"
 :rdd-training-approach :direct :repartition-data :always
 :repartition-strategy :balanced :seed 1234 :save-updater? true
 :storage-level :none))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 3, create a Spark Multi Layer Network
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def your-spark-context
  (new-java-spark-context :app-name "example app"))

;; new-java-spark-context will turn an existing spark-configuration into a java spark context
;; or create a new java spark context with master set to "local[*]" and the app name
;; set to :app-name


(def spark-mln
  (spark-mln/new-spark-multi-layer-network
   :spark-context your-spark-context
   :mln multi-layer-model
   :training-master training-master))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 4, load your data
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; one way is via a dataset-iterator
;; can make one directly from a dataset (iterator data-set)
;; see: nd4clj.linalg.dataset.api.data-set and nd4clj.linalg.dataset.data-set
;; we are going to use a pre-built one

(def iris-iter (new-iris-data-set-iterator :batch-size 1 :n-examples 5))

;; now lets convert the data into a javaRDD

(def our-rdd (java-rdd-from-iter :spark-context your-spark-context :iter iris-iter))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step 5, fit and evaluate the model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def fitted-spark-mln (fit-spark-mln! :spark-mln spark-mln :rdd our-rdd))
;; this fn also has the option to supply :path-to-data instead of :rdd
;; that path should point to a directory containing a number of dataset objects

(def eval-obj (eval-classification-spark-mln :spark-mln fitted-spark-mln
                                             :rdd our-rdd))
;; we would want to have different testing and training rdd's but here we are using
;; the data we trained on

;; lets get the stats for how our model performed

(get-stats :evaler eval-obj)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; It is also possible to train single layer models via spark
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def single-layer-model
  (nn-conf/nn-conf-builder
   :optimization-algo :stochastic-gradient-descent
   :learning-rate 0.006
   :build? true
   :layer (l/dense-layer-builder :n-in 10 :n-out 2 :activation-fn :relu)))

(def spark-layer (new-spark-dl4j-layer :spark-context your-spark-context
                                       :nn-conf single-layer-model))

(def fitted-spark-layer (fit-spark-layer-with-ds! :rdd our-rdd
                                                  :spark-layer spark-layer))

;; need to figure out creating these org.apache.spark.mllib.linalg.Matrix or org.apache.spark.mllib.linalg.Vector
;; needed to call predict

;; also need to figure out JavaRDD<org.nd4j.linalg.dataset.DataSet> -> JavaRDD<org.apache.spark.mllib.regression.LabeledPoint>
;; this is for single fn call training (fit-spark-layer!...)
```

## NOTES

There are 3 types of arg structures for fns found in this libarary
- single arg, just pass the arg to the function

- single keyword arg, used when a function can accept either a single arg or no args

- multiple keyword args, used when a function expects many args
  - the function may expect all args to be supplied
  - the function may expect cominations of args
     - different combinations will produce different results

 - see function defns to determine how the function behaves
   - doc strings describe the arg types and in some places the result of various combinations of args
     - I plan on adding the result of various cominations along with the arg descriptions
   - there are cases when the function can accept all args or a subset of them

- specs are going to replace the assertions made within the function definitions
  - eventually all fns will be spec'd

The namespaces contain dl4j user facing functions and functions that are called behind the scene
- this decision was to allow for future development upon completion of the wrapping
- this also allows for experimenting and fine grane control over the deeplearning building process
- look at the dl4j source for a better understanding of the wrapped code
  - links to the java-docs are in some name spaces and not other
    - eventually all name spaces will have refrences to the dl4j java docs

##Terminology

Coming soon


## TODO

Finish tests of currently implemented classes/interfaces
- dl4clj.nn.layers.feedforward.recursive.tree

Refactor overall structure of this project
- ensure consistency in stlye (multimethods for heavy lifting and fns for use)
- ensure no cascading config maps
- seperation of interfaces from the classes/namespaces that implement/use them
- general refinement

Improve examples
- minimal importing
- replace method calls with fn calls

Update doc strings
- content and formatting

Fix built-in dl4j logging

ND4j and Datavec implementations
- some of this has already been done but is no where close to complete

Update release section

Spark and Kafka streaming

Parallelism (single machine)

## Packages to implement:

Implement ComputationGraphs and the classes which use them
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/graph/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ComputationGraphConfiguration.GraphBuilder.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ComputationGraphConfiguration.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/graph/package-frame.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/graph/rnn/package-frame.html>

Storage
- <https://deeplearning4j.org/doc/org/deeplearning4j/api/storage/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/api/storage/impl/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/api/storage/listener/RoutingIterationListener.html>

AWS
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/dataset/DataSetLoader.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/ec2/Ec2BoxCreator.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/ec2/provision/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/s3/BaseS3.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/s3/reader/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/aws/s3/uploader/S3Uploader.html>

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

Kuromoji (https://github.com/atilika/kuromoji)
self-contained and very easy to use Japanese morphological analyzer designed for search
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/buffer/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/compile/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/dict/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/io/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/ipadic/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/ipadic/compile/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/trie/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/util/package-summary.html>
- <https://deeplearning4j.org/doc/com/atilika/kuromoji/viterbi/package-summary.html>

Keras
- <https://deeplearning4j.org/doc/org/deeplearning4j/keras/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/layers/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/preprocessors/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/trainedmodels/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/trainedmodels/Utils/package-summary.html>

Berkeley
- <https://deeplearning4j.org/doc/org/deeplearning4j/berkeley/package-summary.html>

RecordMetaData
- <https://deeplearning4j.org/doc/org/deeplearning4j/eval/meta/Prediction.html>

MNIST (not sure if necessary)
- <https://deeplearning4j.org/doc/org/deeplearning4j/datasets/mnist/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/datasets/mnist/draw/package-summary.html>


Parallelism
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/factory/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/main/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/parameterserver/package-summary.html>
- <https://deeplearning4j.org/doc/org/deeplearning4j/parallelism/trainer/package-summary.html>

TSNE
- <https://deeplearning4j.org/doc/org/deeplearning4j/plot/package-summary.html>

Utils
- <https://deeplearning4j.org/doc/org/deeplearning4j/util/package-summary.html>

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


*Drops the mic*
