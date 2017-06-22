(ns ^{:doc "Interface for saving MultiLayerNetworks learned during early stopping, and retrieving them again later

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/EarlyStoppingModelSaver.html"}
    dl4clj.earlystopping.interfaces.model-saver
  (:import [org.deeplearning4j.earlystopping EarlyStoppingModelSaver])
  (:require [dl4clj.earlystopping.model-saver :refer [model-saver-type]]))

(defn get-best-model
  "Retrieve the best model that was previously saved

  saver (map or obj), either an object created by one of the new-saver fns
   or a config map for calling model-saver-type
   - see: dl4clj.earlystopping.model-saver"
  [saver]
  ;; there is also a computation graph saver but computation graphs are not implemented
  (.getBestModel (if (map? saver)
                   (model-saver-type saver)
                   saver)))

(defn get-latest-model
  "Retrieve the most recent model that was previously saved

  saver (map or obj), either an object created by one of the new-saver fns
   or a config map for calling model-saver-type
   - see: dl4clj.earlystopping.model-saver"
  [saver]
  (.getLatestModel (if (map? saver)
                     (model-saver-type saver)
                     saver)))

(defn save-best-model!
  "Save the best model (so far) learned during early stopping training

  :saver (map or obj), either an object created by one of the new-saver fns
   or a config map for calling model-saver-type
   - see: dl4clj.earlystopping.model-saver

  :net (nn), a neural network

  :score (double), the score for the nn

  returns the saver"
  [& {:keys [saver net score]}]
  (let [s (if (map? saver)
            (model-saver-type saver)
            saver)]
    (doto s (.saveBestModel net score))))

(defn save-latest-model!
  "Save the latest (most recent) model learned during early stopping

  :saver (map or obj), either an object created by one of the new-saver fns
   or a config map for calling model-saver-type
   - see: dl4clj.earlystopping.model-saver

  :net (nn), a neural network

  :score (double), the score for the nn

  returns the saver"
  [& {:keys [saver net score]}]
  (let [s (if (map? saver)
            (model-saver-type saver)
            saver)]
   (doto s (.saveLatestModel net score))))
