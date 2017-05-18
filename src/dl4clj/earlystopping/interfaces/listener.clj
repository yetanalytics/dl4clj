(ns ^{:doc "EarlyStoppingListener is a listener interface for conducting early stopping training.
 It provides onStart, onEpoch, and onCompletion methods, which are called as appropriate

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/listener/EarlyStoppingListener.html"}
    dl4clj.earlystopping.interfaces.listener
  (:import [org.deeplearning4j.earlystopping.listener EarlyStoppingListener]))

(defn on-completion!
  "Method that is called at the end of early stopping training

  :early-stopping-result (es-result)
   - The early stopping result. Provides details of why early stopping training was terminated, etc

   - see: TBD"
  [& {:keys [listener early-stopping-result]}]
  (doto listener (.onCompletion early-stopping-result)))

(defn on-epoch!
  "Method that is called at the end of each epoch completed during early stopping training

  :epoch-n (int), The number of the epoch just completed (starting at 0)

  :score (double), The score calculated

  :early-stop-conf (es-conf), The configuration for the early stopping
   - see: dl4clj.earlystopping.early-stopping-config

  :nn (neural network), a built neural network
   - see: dl4clj.nn.conf.builders.nn-conf-builder"
  [& {:keys [listener epoch-n score early-stop-conf nn]}]
  (doto listener (.onEpoch epoch-n score early-stop-conf nn)))

(defn on-start!
  "Method to be called when early stopping training is first started

  :early-stop-conf (es-conf), The configuration for the early stopping
   - see: dl4clj.earlystopping.early-stopping-config

  :nn (neural network), a built neural network
   - see: dl4clj.nn.conf.builders.nn-conf-builder"
  [& {:keys [listener early-stop-conf nn]}]
  (doto listener (.onStart early-stop-conf nn)))
