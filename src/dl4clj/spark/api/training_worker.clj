(ns ^{:doc "TrainingWorker is a small serializable class that can be passed (in serialized form) to each Spark executor for actually conducting training. The results are then passed back to the TrainingMaster for processing.

TrainingWorker implementations provide a layer of abstraction for network learning tha should allow for more flexibility/ control over how learning is conducted (including for example asynchronous communication)

see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/api/TrainingWorker.html"}
    dl4clj.spark.api.training-worker
  (:import [org.deeplearning4j.spark.api TrainingWorker]))

;; param-avg-worker currently only implementer

(defn add-hook!
  "Add a training hook to be used during training of the worker

  returns the worker (currently only a param averaging worker)"
  [& {:keys [worker training-hook]}]
  (doto worker (.addHook training-hook)))

(defn remove-hook!
  "removes a training hook from the worker

  returns the worker (currently only a param averaging worker)"
  [& {:keys [worker training-hook]}]
  (doto worker (.removeHook training-hook)))

(defn get-data-config
  "returns the worker config that contains info such as minibatch size"
  [worker]
  (.getDataConfiguration worker))

(defn get-final-result
  "Get the final result to be returned to the driver

  :with-stats? (boolean), defaults to false
   - should be set to true when spark training stats are being collected"
  ;; this method can also take a comp graph
  [& {:keys [worker mln with-stats?]
      :or {with-stats? false}}]
  (if (true? with-stats?)
    (.getFinalResultWithStats worker mln)
    (.getFinalResult worker mln)))

(defn get-final-result-no-data
  "Get the final result to be returned to the driver, if no data was available for this executor

  :with-stats? (boolean), defaults to false
   - should be set to true when spark training stats are being collected"
  [& {:keys [worker with-stats?]
      :or {with-stats? false}}]
  (if (true? with-stats?)
    (.getFinalResultNoDataWithStats worker)
    (.getFinalResultNoData worker)))

(defn get-initial-model
  "Get the initial model when training a MultiLayerNetwork/SparkDl4jMultiLayer"
  [worker]
  (.getInitialModel worker))

(defn process-mini-batch!
  "Process (fit) a minibatch for a MultiLayerNetwork

  :data-set (ds), the dataset to train on

  :mln (nn), the multi layer network to train

  :is-last? (boolean), is this the last dataset or will more be processed after this one

  :with-stats? (boolean), defaults to false
   - should be set to true when spark training stats are being collected

  returns a map of the mln and the worker when with-stats? is false
  otherwise returns a pair containing the stats"
  [& {:keys [data-set mln is-last? with-stats? worker]
      :or {with-stats? false}}]
  (if (true? with-stats?)
    (.processMinibatchWithStats worker data-set mln is-last?)
    (do (.processMinibatch worker data-set mln is-last?)
        {:mln mln :worker worker})))
