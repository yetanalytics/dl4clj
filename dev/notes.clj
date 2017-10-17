(ns dev.notes
  (:require [dl4clj.nn.api.model :as model]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.datasets.iterators :as iter]
            [dl4clj.eval.evaluation :refer [new-classification-evaler]]
            [dl4clj.eval.api.eval :refer [eval-classification! get-accuracy]]
            [dl4clj.nn.multilayer.multi-layer-network :as mln]))

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
   :pretrain? false
   ;; we want the dl4j object
   :as-code? false))

(dl4clj.nn.api.multi-layer-network/is-init-called?
 (dl4clj.nn.api.multi-layer-network/initialize! :mln (mln/new-multi-layer-network :conf nn-conf)
                                                :ds (dl4clj.datasets.new-datasets/new-ds :input [1 2 3 4]
                                                                                         :output [1]
                                                                                         :as-code? false)))

;; need to make a note that you can initialize a model by calling model/init or multi-layer-network/initialize
;; init does not require any parameters where as initiailze requires a dataset

;;public void initialize(DataSet data)
;;{
;;   setInput(data.getFeatureMatrix());
;;   feedForward(getInput());
;;   this.labels = data.getLabels();
;;   if (getOutputLayer() instanceof IOutputLayer) {
;;                                                  IOutputLayer ol = (IOutputLayer) getOutputLayer();
;;                                                  ol.setLabels(labels);
;;                                                  }
;;   }

(def mln (model/init! :model (mln/new-multi-layer-network :conf nn-conf)))

;; evaluation given just a model and an iter
;; returns an eval object

(def train-mnist-iter (iter/new-mnist-data-set-iterator :batch-size 10 :train? true
                                                        :seed 123 :as-code? false))

(dl4clj.eval.api.eval/get-stats
 :evaler
 (dl4clj.nn.api.multi-layer-network/evaluate-classification :mln mln :iter train-mnist-iter))

;; evaluation given an evaler, a model and data
;; requires the creation of an evaluation object

(def example-evaler-obj (new-classification-evaler :n-classes 10 :as-code? false))

;; this wont work as is
#_(eval-classification! :evaler example-evaler-obj :mln mln
                        :labels [] :features [])

;; ns for fit fns


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;dl4clj.eval.api.eval
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; eval-classification!
;; eval-time-series!
;; based on the other wrapper, these happen behind the scenes
;; should only need the evaluate fns/methods bellow

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;dl4clj.nn.api.multi-layer-network
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; evaluate-classification
;; evaluate-regression
;; evaluate-roc
;; evaluate-roc-multi-class

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;dl4clj.spark.api.dl4j-multi-layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; eval-classification-spark-mln
;; eval-regression-spark-mln
;; eval-roc-spark-mln
;; eval-multi-class-roc-spark-mln
