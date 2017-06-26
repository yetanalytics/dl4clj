(ns dl4clj.dev.usr
  (:require [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn-conf]
            [dl4clj.nn.conf.builders.builders :as l]
            ;;[clj-time.core :as t]
            ;;[clj-time.format :as tf]
            [datavec.api.split :as f]
            [datavec.api.records.readers :as rr]
            [dl4clj.datasets.datavec :as ds]
            [nd4clj.linalg.dataset.api.data-set :as d]
            [dl4clj.eval.evaluation :as e]
            [dl4clj.nn.multilayer.multi-layer-network :as mln])
  (:import [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.datavec.api.transform.schema Schema Schema$Builder]
           [org.datavec.api.transform TransformProcess$Builder TransformProcess]
           [org.apache.spark SparkConf]
           [org.apache.spark.api.java JavaRDD JavaSparkContext]
           [org.datavec.spark.transform SparkTransformExecutor]
           [org.datavec.spark.transform.misc StringToWritablesFunction]
           [java.util.List]
           ))

;;TODO

;;testing data https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data
;;training data https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data
;;desc https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand.names
;; attributes in data
;; pairs of suit, rank of card (5 cards = 10 pairs)
;; last value is class of poker hand
;; 0-9

;; set up the evaluator
(comment


  (defn timestamp-now
    "Returns a string timestamp in the form of 2015-10-02T18:27:43Z."
    []
    (str (tf/unparse (tf/formatters :date-hour-minute-second)
                     (t/now))
         "Z"))

  (defn initialize-record-reader
    [record-reader file-path]
    (doto record-reader
      (rr/initialize (f/new-filesplit {:root-dir file-path}))))

  (defn set-up-data-set-iterator
    [record-reader batch-size label-idx num-diff-labels]
    (ds/iterator {:rr-dataset-iter {:record-reader record-reader
                                    :batch-size batch-size
                                    :label-idx label-idx
                                    :n-possible-labels num-diff-labels}}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; import the data
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (def fresh-csv-rr (rr/record-reader {:csv-rr {}}))

  (def initialized-training-rr
    (initialize-record-reader fresh-csv-rr "resources/poker/poker-hand-training.csv"))

  (def initialized-testing-rr
    (initialize-record-reader fresh-csv-rr "resources/poker/poker-hand-testing.csv"))

  (def training-iter
    (set-up-data-set-iterator initialized-training-rr 25 10 10))

  (def testing-iter
    (set-up-data-set-iterator initialized-testing-rr 25 10 10))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; set up network
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (def conf
    ;; play around with regularization (L1, L2, dropout ...)
    (->
     (.build
      (nn-conf/nn-conf-builder {:seed 123
                                :optimization-algo :stochastic-gradient-descent
                                :iterations 1
                                :learning-rate 0.006
                                :updater :nesterovs
                                :momentum 0.9
                                :pre-train false
                                :backprop true
                                :layers {0 {:dense-layer {:n-in 10
                                                          :n-out 30
                                                          :weight-init :xavier
                                                          :activation-fn :relu}}
                                         1 {:output-layer {:n-in 30
                                                           :loss-fn :negativeloglikelihood
                                                           :weight-init :xavier
                                                           :activation-fn :soft-max
                                                           :n-out 10}}}}))
     (mlb/multi-layer-config-builder {})))

  (def model (mln/multi-layer-network conf))

  (defn init [mln]
    ;;.init is going to be implemented in dl4clj.nn.multilayer.multi-layer-network
    (doto mln
      .init))

  (def init-model (init model))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; train-model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (def numepochs 10)

  (defn ex-train [mln]
    (loop
        [i 0
         result {}]
      (cond (not= i numepochs)
            (do
              (println "current at epoch:" i)
              (recur (inc i)
                     (.fit mln training-iter)))
            ;;.fit is going to be implemented in dl4clj.nn.multilayer.multi-layer-network
            (= i numepochs)
            (do
              (println "training done")
              mln))))

  (def trained-model (ex-train init-model))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; evaluate model
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (def evaler-c (e/new-evaluator {:n-classes 10}))

  (defn eval-model [mln]
    (while (true? (.hasNext testing-iter))
      (let [nxt (.next testing-iter)
            output (.output mln (.getFeatureMatrix nxt))]
        (do (.eval e (.getLabels nxt) output)
            (println (.stats e))))))

  #_(defn eval-model-classification
      [mln testing-iter evaler]
      (while (true? (rr/has-next? testing-iter))
        (let [next (rr/next-data-record testing-iter)
              output (.output mln (d/get-feature-matrix nxt))]
          ;;.output is going to be implemented in dl4clj.nn.multilayer.multi-layer-network
          (do (e/eval-classification
               evaler
               (rr/get-labels next)
               output)
              (println (e/get-stats evaler))))
        ))

  (def evaled-model (eval-model trained-model))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; graveyard
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


  (def num-lines-to-skip 0)
  (def delim ",")
  (def base-dir "/Users/will/projects/dl4clj/resources/")
  (def file-name "poker-hand-training.txt")
  (def input-path (str base-dir file-name))
  (def output-path (str base-dir "reports_processed_" (timestamp-now)))

  (def input-schema-old
    (.build
     (doto (Schema$Builder. )
       (.addColumnCategorical "suit-c1" '("1" "2" "3" "4"))
       (.addColumnLong "card-rank-1" 1 13) ;;first card
       (.addColumnCategorical "suit-c2" '("1" "2" "3" "4"))
       (.addColumnLong "card-rank-2" 1 13) ;; second card
       (.addColumnCategorical "suit-c3" '("1" "2" "3" "4"))
       (.addColumnLong "card-rank-3" 1 13) ;; third card
       (.addColumnCategorical "suit-c4" '("1" "2" "3" "4"))
       (.addColumnLong "card-rank-4" 1 13) ;; 4th card
       (.addColumnCategorical "suit-c5" '("1" "2" "3" "4"))
       (.addColumnLong "card-rank-5" 1 13) ;; 5th card
       (.addColumnCategorical "hand-value" '("0" "1" "2" "3" "4" "5" "6" "7" "8" "9"))
       )))

  (def experimental-schema
    (.build
     (doto (Schema$Builder.)
       (.addColumnsLong "everything" 0 11))))

  (def data-transform
    (.build
     (doto (TransformProcess$Builder. experimental-schema)
       (.categoricalToInteger (into-array String "everything") #_(into-array String '("suit-c1" "suit-c2" "suit-c3"
                                                                                      "suit-c4" "suit-c5" "hand-value"))))))

  (def experimental-da)

  (defn check-data-transform
    [dt]
    (let [num-actions (.size (.getActionList dt))]
      (loop [i 0]
        (cond (not= i num-actions)
              (do
                (println (str "Step " i ": " (.get (.getActionList dt) i) "\n"))
                (println (str "This is what the data now looks like: "
                              (.getSchemaAfterStep dt i)))
                (recur (inc i)))
              (= i num-actions)
              (println "no more transforms")))))

  (check-data-transform data-transform)
  (.get (.getActionList data-transform) 0)
  #_(def new-spark-conf (SparkConf. ))

  #_(def spark-conf
      (doto new-spark-conf
        (.setMaster "local[*]")
        (.setAppName "poker hand classification")))

  #_(def spark-context (JavaSparkContext. spark-conf))

  #_(.textFile spark-context "resources/poker-spark-test.csv")

  ;; use spark to read the data
  (def lines (.textFile spark-context input-path))

  #_(.textFile spark-context input-path)

  #_(.textFile spark-context )
  ;; convert to writable

  ;; need our writeable fn
  (def writeable-fn (StringToWritablesFunction. (CSVRecordReader.)))

  (def poker-hands
    (doto lines
      (.map writeable-fn)))
  (.map  lines (StringToWritablesFunction. (CSVRecordReader.)))
  ( poker-hands)
  (type (Strun-ringToWritablesFunction. (CSVRecordReader.)))
  (type poker-hands)
  (type data-transform)

  (.getFinalSchema data-transform)
  (.getActionList data-transform)
  (def processed (SparkTransformExecutor/execute poker-hands data-transform))
  (SparkTransformExecutor/executeSequenceToSeparate poker-hands data-transform)
  (.first poker-hands)
  (.execute  poker-hands (.build
                          (.categoricalToInteger
                           (TransformProcess$Builder.
                            (.build
                             (-> (Schema$Builder. ))

                             (.addColumnCategorical "suit-c1" '("1" "2" "3" "4"))))
                           (into-array String (list "suit-c1")))))
  (.executeToSequence poker-hands data-transform)
  )


(comment
  "java still faster :()"
 (use '[dl4clj.nn.conf.builders.nn-conf-builder])
(use '[dl4clj.nn.conf.builders.multi-layer-builders])
(use '[dl4clj.datasets.iterator.impl.default-datasets])
(use '[dl4clj.optimize.listeners.listeners])
(use '[dl4clj.nn.multilayer.multi-layer-network])
(use '[dl4clj.nn.api.model])
(use '[dl4clj.helpers])

(def train-mnist-iter (new-mnist-data-set-iterator :batch-size 64 :train? true :seed 123))

#_(reset-if-not-at-start! train-mnist-iter)

(def lazy-l-builder (nn-conf-builder
                :optimization-algo :stochastic-gradient-descent
                :seed 123 :iterations 1 :default-activation-fn :relu
                :regularization? true :default-l2 7.5e-6
                :default-weight-init :xavier :default-learning-rate 0.0015
                :default-updater :nesterovs :default-momentum 0.98
                :layers {0 {:dense-layer
                            {:layer-name "example first layer"
                             :n-in 784 :n-out 500}}
                         1 {:dense-layer
                            {:layer-name "example second layer"
                             :n-in 500 :n-out 100}}
                         2 {:output-layer
                            {:n-in 100 :n-out 10 :loss-fn :negativeloglikelihood
                             :activation-fn :softmax
                             :layer-name "example output layer"}}}))

(def lazy-multi-layer-conf
  (multi-layer-config-builder
   :list-builder lazy-l-builder
   :backprop? true
   :pretrain? false))

(def multi-layer-network-lazy-training
  (init! :model (new-multi-layer-network :conf lazy-multi-layer-conf)))

(def lazy-score-listener (new-score-iteration-listener :print-every-n 5))


(def mln-lazy-with-listener (set-listeners! :model multi-layer-network-lazy-training
                                            :listeners [lazy-score-listener]))

(def lazy-data (data-from-iter train-mnist-iter))


#_(time
 (train-mln-with-lazy-seq! :lazy-seq-data lazy-data :mln mln-lazy-with-listener
                           :n-epochs 15))

;; => "Elapsed time: 262699.667516 msecs"






(def l-builder (nn-conf-builder
                     :optimization-algo :stochastic-gradient-descent
                     :seed 123 :iterations 1 :default-activation-fn :relu
                     :regularization? true :default-l2 7.5e-6
                     :default-weight-init :xavier :default-learning-rate 0.0015
                     :default-updater :nesterovs :default-momentum 0.98
                     :layers {0 {:dense-layer
                                 {:layer-name "example first layer"
                                  :n-in 784 :n-out 500}}
                              1 {:dense-layer
                                 {:layer-name "example second layer"
                                  :n-in 500 :n-out 100}}
                              2 {:output-layer
                                 {:n-in 100 :n-out 10 :loss-fn :negativeloglikelihood
                                  :activation-fn :softmax
                                  :layer-name "example output layer"}}}))

(def multi-layer-conf
  (multi-layer-config-builder
   :list-builder l-builder
   :backprop? true
   :pretrain? false))

(def multi-layer-network-standard-training
  (init! :model (new-multi-layer-network :conf multi-layer-conf)))


(def score-listener (new-score-iteration-listener :print-every-n 5))

(def mln-standard-with-listener (set-listeners! :model multi-layer-network-standard-training
                                                :listeners [score-listener]))

#_(def trained-mln (time (train-mln-with-ds-iter! :mln mln-standard-with-listener
                                          :ds-iter train-mnist-iter
                                          :n-epochs 15)))
;; => "Elapsed time: 220454.276825 msecs"
)
