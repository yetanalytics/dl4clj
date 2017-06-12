(ns dl4clj.streaming.kafka.ndarray-publisher
  (:import [org.deeplearning4j.streaming.kafka NDArrayPublisher]))

#_(NDArrayPublisher. )
;; another broken consturctor

(defn publish!
  "publish an INDArray, can be a single array or an array of arrays"
  [& {:keys [ind-array publisher]}]
  (doto publisher (.publish ind-array)))

(defn start-publisher!
  "starts the publisher"
  [publisher]
  (doto publisher (.start)))
