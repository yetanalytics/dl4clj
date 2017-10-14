(ns ^{:doc "Interface for saving MultiLayerNetworks learned during early stopping, and retrieving them again later

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/EarlyStoppingModelSaver.html"}
    dl4clj.earlystopping.api.model-saver
  (:import [org.deeplearning4j.earlystopping EarlyStoppingModelSaver])
  (:require [dl4clj.earlystopping.model-saver :refer [model-saver-type]]
            [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code?]]))

(defn get-best-model
  "Retrieve the best model that was previously saved

  saver (obj), an object created by one of the new-saver fns and used within
   an early stopping config
   - see: dl4clj.earlystopping.model-saver and dl4clj.earlystopping.early-stopping-config
   - this fn should be called on the saver object after the early-stopping-trainer has been fit"
  [saver & {:keys [as-code?]
            :or {as-code? true}}]
  (match [saver]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getBestModel ~saver))
         :else
         (.getBestModel saver)))

(defn get-latest-model
  "Retrieve the most recent model that was previously saved

  saver (map or obj), either an object created by one of the new-saver fns
   or a config map for calling model-saver-type
   - see: dl4clj.earlystopping.model-saver
   - this fn should be called on the saver object after the early-stopping-trainer has been fit"
  [saver & {:keys [as-code?]
            :or {as-code? true}}]
  (match [saver]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLatestModel ~saver))
         :else
         (.getLatestModel saver)))
