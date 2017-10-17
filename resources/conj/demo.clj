(require '[dl4clj.datasets.iterators :as iter]
         '[dl4clj.datasets.input-splits :as is]
         '[dl4clj.datasets.record-readers :as rr]
         '[dl4clj.datasets.api.record-readers
           :refer [initialize-rr!]]
         '[dl4clj.datasets.pre-processors
           :refer [new-standardize-normalization-ds-preprocessor]]
         '[dl4clj.datasets.api.iterators
           :refer [next-example! get-pre-processor set-pre-processor!]]
         '[dl4clj.core :as c]
         '[clojure.pprint :as pp])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;general data import
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def poker-path "resources/poker-hand-training.csv")

(def file-split (is/new-filesplit :path poker-path))

(def csv-rr
  (initialize-rr!
   :rr (rr/new-csv-record-reader
        :skip-n-lines 0
        :delimiter ",")
   :input-split file-split))

(def csv-iter
  (iter/new-record-reader-dataset-iterator
   :record-reader csv-rr
   :batch-size 1
   :label-idx 10
   :n-possible-labels 10))

(def normalized-iter-obj
  (c/normalize-iter!
   :iter csv-iter
   :normalizer (new-standardize-normalization-ds-preprocessor)
   :as-code? false))

(println (str (next-example! normalized-iter-obj)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Hello world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(require '[dl4clj.nn.conf.builders.nn :as nn]
         '[dl4clj.earlystopping.termination-conditions
           :as term-cond]
         '[dl4clj.nn.api.multi-layer-network
           :refer [evaluate-classification]]
         '[dl4clj.eval.api.eval :refer [get-stats get-accuracy]]
         '[dl4clj.utils :as u])

;; conf

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

;; data

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

;; termination conditions

(def invalid-score-condition
  (term-cond/new-invalid-score-iteration-termination-condition))

(def max-number-epochs-condition
  (term-cond/new-max-epochs-termination-condition :max-n 2))

(def mln-for-pp
  (c/train-with-early-stopping
   :nn-conf nn-conf
   :training-iter train-mnist-iter
   :testing-iter test-mnist-iter
   :eval-every-n-epochs 1
   :iteration-termination-conditions [invalid-score-condition]
   :epoch-termination-conditions [max-number-epochs-condition]
   :save-last-model? false))

(pp/pprint mln-for-pp)

;; load trained mln

(def trained-mln
  (u/load-model! :path "resources/conj/mnist-model"))

(def evaler
  (evaluate-classification
   :mln trained-mln
   :iter test-mnist-iter))

(println (get-accuracy evaler))

(println (str (get-stats :evaler evaler)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; simple lstm example (time series classification)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(require '[dl4clj.nn.conf.builders.layers :as l]
         '[dl4clj.nn.multilayer.multi-layer-network :as mln]
         '[dl4clj.optimize.listeners :as listener]
         '[dl4clj.nn.api.model :refer [set-listeners!]])

;; where our data lives
(def train-features-location
  "/Users/will/projects/dl4clj/dev-resources/uci/train/features/%d.csv")

(def train-labels-location
  "/Users/will/projects/dl4clj/dev-resources/uci/train/labels/%d.csv")

(def test-features-location
  "/Users/will/projects/dl4clj/dev-resources/uci/test/features/%d.csv")

(def test-labels-location
  "/Users/will/projects/dl4clj/dev-resources/uci/test/labels/%d.csv")

;; same flow from general data import
;; but now using a different type of input split

(def train-features-input-split
  (is/new-numbered-file-input-split
   :base-string train-features-location
   :inclusive-min-idx 0
   :inclusive-max-idx 449))

(def train-labels-input-split
  (is/new-numbered-file-input-split
   :base-string train-labels-location
   :inclusive-min-idx 0
   :inclusive-max-idx 449))

(def test-features-input-split
  (is/new-numbered-file-input-split
   :base-string test-features-location
   :inclusive-min-idx 450
   :inclusive-max-idx 599))

(def test-labels-input-split
  (is/new-numbered-file-input-split
   :base-string test-labels-location
   :inclusive-min-idx 450
   :inclusive-max-idx 599))

;; create and load data into our record reader

(def train-features-seq-rr
  (initialize-rr!
   :rr (rr/new-csv-seq-record-reader)
   :input-split train-features-input-split))

(def train-labels-seq-rr
  (initialize-rr!
   :rr (rr/new-csv-seq-record-reader)
   :input-split train-labels-input-split))

(def test-features-seq-rr
  (initialize-rr!
   :rr (rr/new-csv-seq-record-reader)
   :input-split test-features-input-split))

(def test-labels-seq-rr
  (initialize-rr!
   :rr (rr/new-csv-seq-record-reader)
   :input-split test-labels-input-split))

;; create our iterators from the record readers

(def train-iter
  (iter/new-seq-record-reader-dataset-iterator
   :features-reader train-features-seq-rr
   :labels-reader train-labels-seq-rr
   :mini-batch-size 10
   :n-possible-labels 6
   :regression? false
   :alignment-mode :align-end))

(def test-iter
  (iter/new-seq-record-reader-dataset-iterator
   :features-reader test-features-seq-rr
   :labels-reader test-labels-seq-rr
   :mini-batch-size 10
   :n-possible-labels 6
   :regression? false
   :alignment-mode :align-end))

;; normalize our data

(def normalized-train-iter
  (c/normalize-iter!
   :iter train-iter
   :normalizer (new-standardize-normalization-ds-preprocessor)))

(def fitted-normalizer
  (get-pre-processor normalized-train-iter))

(def normalized-test-iter
  (set-pre-processor!
   :iter test-iter
   :pre-processor fitted-normalizer))

;; mln configuration

(def mln-conf
  (nn/builder
   :seed 123
   :optimization-algo :stochastic-gradient-descent
   :default-weight-init :xavier
   :default-updater :nesterovs
   :default-learning-rate 0.005
   :default-gradient-normalization :clip-element-wise-absolute-value
   :default-gradient-normalization-threshold 0.5
   :pretrain? false
   :backprop? true
   ;; can pass layers as config map or fns
   :layers {0 (l/graves-lstm-layer-builder
               :activation-fn :tanh
               :n-in 1
               :n-out 10)
            1 (l/rnn-output-layer-builder
               :loss-fn :mcxent
               :activation-fn :softmax
               :n-in 10
               :n-out 6)}))

(def lstm-model
  (c/model-from-conf :nn-conf mln-conf))

(pp/pprint lstm-model)

(def trained-lst-model
  (u/load-model! :path "resources/conj/lstm-model/"))

(def lstm-evaler
  (evaluate-classification :mln trained-lst-model
                           :iter normalized-test-iter))

(println (str (get-stats :evaler lstm-evaler)))
