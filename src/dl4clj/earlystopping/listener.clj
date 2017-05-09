(ns dl4clj.earlystopping.listener
  (:import [org.deeplearning4j.earlystopping.listener EarlyStoppingListener]))

(defn on-completion!
  [& {:keys [listener early-stopping-result]}]
  (doto listener (.onCompletion early-stopping-result)))

(defn on-epoch!
  [& {:keys [listener epoch-n score early-stop-conf nn]}]
  (doto listener (.onEpoch epoch-n score early-stop-conf nn)))

(defn on-start!
  [& {:keys [listener early-stop-conf nn]}]
  (doto listener (.onStart early-stop-conf nn)))
