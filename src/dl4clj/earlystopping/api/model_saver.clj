(ns ^{:doc "Interface for saving MultiLayerNetworks learned during early stopping, and retrieving them again later

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/EarlyStoppingModelSaver.html"}
    dl4clj.earlystopping.api.model-saver
  (:import [org.deeplearning4j.earlystopping EarlyStoppingModelSaver])
  (:require [dl4clj.earlystopping.model-saver :refer [model-saver-type]]))

(defn get-best-model
  "Retrieve the best model that was previously saved

  saver (obj), an object created by one of the new-saver fns and used within
   an early stopping config
   - see: dl4clj.earlystopping.model-saver and dl4clj.earlystopping.early-stopping-config
   - this fn should be called on the saver object after the early-stopping-trainer has been fit"
  [saver]
  ;; there is also a computation graph saver but computation graphs are not implemented
  (.getBestModel saver))

(defn get-latest-model
  "Retrieve the most recent model that was previously saved

  saver (map or obj), either an object created by one of the new-saver fns
   or a config map for calling model-saver-type
   - see: dl4clj.earlystopping.model-saver
   - this fn should be called on the saver object after the early-stopping-trainer has been fit"
  [saver]
  (.getLatestModel saver))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; implementation fns, will be removed in the future master branch but kept
;; in a dev branch (moved to a dev ns)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn save-best-model!
  "Save the best model (so far) learned during early stopping training
   - fn interal to dl4j, should not need to be called by a user

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
  - fn interal to dl4j, should not need to be called by a user

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
