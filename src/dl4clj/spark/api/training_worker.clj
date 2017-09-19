(ns ^{:doc "TrainingWorker is a small serializable class that can be passed (in serialized form) to each Spark executor for actually conducting training. The results are then passed back to the TrainingMaster for processing.

TrainingWorker implementations provide a layer of abstraction for network learning tha should allow for more flexibility/ control over how learning is conducted (including for example asynchronous communication)

see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/api/TrainingWorker.html"}
    dl4clj.spark.api.training-worker
  (:import [org.deeplearning4j.spark.api TrainingWorker])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code?]]))

;; param-avg-worker currently only implementer

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-data-config
  "returns the worker config that contains info such as minibatch size"
  [& {:keys [worker as-code?]
      :or {as-code? true}}]
  (match [worker]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getDataConfiguration ~worker))
         :else
         (.getDataConfiguration worker)))

(defn get-final-result
  "Get the final result to be returned to the driver"
  ;; this method can also take a comp graph
  [& {:keys [worker mln as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:worker (_ :guard seq?)
           :mln (_ :guard seq?)}]
         (obj-or-code? as-code? `(.getFinalResult ~worker ~mln))
         :else
         (.getFinalResult worker mln)))

(defn get-final-result-with-stats
  "Get the final result to be returned to the driver

  used when spark training stats are being collected"
  [& {:keys [worker mln as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:worker (_ :guard seq?)
           :mln (_ :guard seq?)}]
         (obj-or-code? as-code? `(.getFinalResultWithStats ~worker ~mln))
         :else
         (.getFinalResultWithStats worker mln)))

(defn get-final-result-no-data
  "Get the final result to be returned to the driver, if no data was available for this executor"
  [& {:keys [worker as-code?]
      :or {as-code? true}}]
  (match [worker]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getFinalResultNoData ~worker))
         :else
         (.getFinalResultNoData worker)))

(defn get-final-result-no-data-with-stats
  "Get the final result to be returned to the driver, if no data was available for this executor

   should be used when spark training stats are being collected"
  [& {:keys [worker as-code?]
      :or {as-code? true}}]
  (match [worker]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getFinalResultNoDataWithStats ~worker))
         :else
         (.getFinalResultNoDataWithStats worker)))

(defn get-initial-model
  "Get the initial model when training a MultiLayerNetwork/SparkDl4jMultiLayer"
  [& {:keys [worker as-code?]
      :or {as-code? true}}]
  (match [worker]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getInitialModel ~worker))
         :else
         (.getInitialModel worker)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn add-hook!
  "Add a training hook to be used during training of the worker

  returns the worker (currently only a param averaging worker)"
  [& {:keys [worker training-hook as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:worker (_ :guard seq?)
           :training-hook (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~worker (.addHook ~training-hook)))
         :else
         (doto worker (.addHook training-hook))))

(defn remove-hook!
  "removes a training hook from the worker

  returns the worker (currently only a param averaging worker)"
  [& {:keys [worker training-hook as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:worker (_ :guard seq?)
           :training-hook (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~worker (.removeHook ~training-hook)))
         :else
         (doto worker (.removeHook training-hook))))

(defn process-mini-batch!
  "Process (fit) a minibatch for a MultiLayerNetwork

  :data-set (ds), the dataset to train on

  :mln (nn), the multi layer network to train

  :is-last? (boolean), is this the last dataset or will more be processed after this one

  returns the worker"
  [& {:keys [data-set mln is-last? worker as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:data-set (_ :guard seq?)
           :mln (_ :guard seq?)
           :is-last? (:or (_ :guard boolean?)
                          (_ :guard seq?))
           :worker (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~worker (.processMinibatch ~data-set ~mln ~is-last?)))
         :else
         (doto worker (.processMinibatch data-set mln is-last?))))

(defn process-mini-batch-with-stats!
  "Process (fit) a minibatch for a MultiLayerNetwork

  :data-set (ds), the dataset to train on

  :mln (nn), the multi layer network to train

  :is-last? (boolean), is this the last dataset or will more be processed after this one

  returns a pair containing the stats"
  [& {:keys [data-set mln is-last? worker as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:worker (_ :guard seq?)
           :data-set (_ :guard seq?)
           :mln (_ :guard seq?)
           :is-last? (:or (_ :guard boolean?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.processMinibatchWithStats ~worker ~data-set ~mln ~is-last?))
         :else
         (.processMinibatchWithStats worker data-set mln is-last?)))
