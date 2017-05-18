(ns ^{:doc "Interface for saving MultiLayerNetworks learned during early stopping, and retrieving them again later

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/EarlyStoppingModelSaver.html"}
    dl4clj.earlystopping.interfaces.model-saver
  (:import [org.deeplearning4j.earlystopping EarlyStoppingModelSaver]))

(defn get-best-model
  "Retrieve the best model that was previously saved

  saver is one of: :in-memory-model-saver, :local-file-model-saver
   - see: TBD"
  [saver]
  ;; there is also a computation graph saver but computation graphs are not implemented
  (.getBestModel saver))

(defn get-latest-model
  "Retrieve the most recent model that was previously saved

  saver is one of: :in-memory-model-saver, :local-file-model-saver
   - see: TBD"
  [saver]
  (.getLatestModel saver))

(defn save-best-model!
  "Save the best model (so far) learned during early stopping training

  :saver (es saver) one of: :in-memory-model-saver, :local-file-model-saver
   - see: TBD

  :net (nn), a neural network

  :score (double), the score for the nn

  returns the saver"
  [& {:keys [saver net score]}]
  (doto saver (.saveBestModel net score)))

(defn save-latest-model
  "Save the latest (most recent) model learned during early stopping

  :saver (es saver) one of: :in-memory-model-saver, :local-file-model-saver
   - see: TBD

  :net (nn), a neural network

  :score (double), the score for the nn

  returns the saver"
  [& {:keys [saver net score]}]
  (doto saver (.saveLatestModel net score)))
