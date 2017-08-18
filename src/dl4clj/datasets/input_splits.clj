(ns dl4clj.datasets.input-splits
  (:import [org.datavec.api.split FileSplit BaseInputSplit CollectionInputSplit
            InputStreamInputSplit ListStringSplit NumberedFileInputSplit
            StringSplit TransformSplit TransformSplit$URITransform]
           [org.datavec.api.io.filters BalancedPathFilter RandomPathFilter]
           [org.datavec.api.io.labels PathLabelGenerator ParentPathLabelGenerator
            PatternPathLabelGenerator])
  (:require [clojure.java.io :as io]
            [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [array-of generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input split multimethod
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti input-split
  "Multimethod that builds an input split based on the supplied type and opts"
  generic-dispatching-fn)

(defmethod input-split :file-split [opts]
  (let [config (:file-split opts)
        {root :path
         rng :rng-seed
         fmt :allow-format
         recursive? :recursive?} config
        a-fmt (array-of :data fmt
                        :java-type java.lang.String)
        a-file (io/as-file root)]
    (match [config]
           [{:path _ :allow-format _ :rng-seed _}]
           (FileSplit. a-file a-fmt rng)
           [{:path _ :allow-format _ :recursive? _}]
           (FileSplit. a-file a-fmt recursive?)
           [{:path _ :allow-format _}]
           (FileSplit. a-file a-fmt)
           [{:path _ :rng-seed _}]
           (FileSplit. a-file rng)
           :else
           (FileSplit. a-file))))

(defmethod input-split :collection-input-split [opts]
  (let [config (:collection-input-split opts)
        coll-of-uris (:collection config)]
    (CollectionInputSplit. coll-of-uris)))

(defmethod input-split :input-stream-input-split [opts]
  (let [config (:input-stream-input-split opts)
        {in :in-stream
         path :file-path} config]
    (if path
      (InputStreamInputSplit. in (io/as-file path))
      (InputStreamInputSplit. in))))

(defmethod input-split :list-string-split [opts]
  (let [config (:list-string-split opts)
        d (:data config)]
    (ListStringSplit. d)))

(defmethod input-split :numbered-file-input-split [opts]
  (let [config (:numbered-file-input-split opts)
        {base-str :base-string
         min-idx :inclusive-min-idx
         max-idx :inclusive-max-idx} config]
    (NumberedFileInputSplit. base-str min-idx max-idx)))

(defmethod input-split :string-split [opts]
  (let [config (:string-split opts)
        d (:data config)]
    (StringSplit. d)))

(defmethod input-split :transform-split [opts]
  (let [config (:transform-split opts)
        {src-split :base-input-split
         t :to-be-replaced
         u :replaced-with} config]
    (TransformSplit/ofSearchReplace src-split t u)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; label generators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-parent-path-label-generator
  "Returns as label the base name of the parent file of the path (the directory).

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/ParentPathLabelGenerator.html"
  []
  (ParentPathLabelGenerator.))

(defn new-pattern-path-label-generator
  "Returns a label derived from the base name of the path.
  Splits the base name of the path with the given regex pattern,
  and returns the patternPosition'th element of the array.

  :pattern (str), a string regex

  :pattern-position (int), where to expect the patern within the file-path

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/PatternPathLabelGenerator.html"
  [& {:keys [pattern pattern-position]
      :as opts}]
  (assert (contains? opts :pattern) "you must supply a regex patern to use this label generator")
  (if pattern-position
    (PatternPathLabelGenerator. pattern pattern-position)
    (PatternPathLabelGenerator. pattern)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; path-filter
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-balanced-path-filter
  "Randomizes the order of paths in an array and removes paths randomly to
  have the same number of paths for each label. Further interlaces the paths
  on output based on their labels, to obtain easily optimal batches for training.

  :rng (java.util.Random), a randomly generated number

  :extensions (collection of string(s)), files to keep

  :label-generator (label-generator), call either new-parent-path-label-generator
   or new-pattern-path-label-generator

  :max-paths (int), max number of paths to return (0 = unlimited)

  :max-labels (int), max number of labels to return (0 = unlimited)

  :min-paths-per-label (int), min number of paths per labels to return

  :max-paths-per-label (int), max number of paths per labels to return
   - (0 = unlimited)

  :labels (collection of strings), file-paths you want to keep
   - empty collection = keep all paths

   If min-paths-per-label > 0, it might return an unbalanced set if the value is
  larger than the number of examples available for the label with the minimum amount.

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/filters/BalancedPathFilter.html"
  [& {:keys [rng extensions label-generator max-paths max-labels
             min-paths-per-label max-paths-per-label labels]
      :or {extensions nil
           max-paths 0
           max-labels 0
           min-paths-per-label 0
           max-paths-per-label 0}}]
  (BalancedPathFilter. rng (array-of :data extensions
                                     :java-type java.lang.String)
                       label-generator max-paths max-labels
                       min-paths-per-label max-paths-per-label
                       (array-of :data labels
                                 :java-type java.lang.String)))

(defn new-random-path-filter
  "Randomizes the order of paths in an array.

  :rng (java.util.Random), a random object used as a seed

  :extensions (collection of strings), file types to keep

  :max-paths (int), max number of paths to return
    - 0 = unlimited

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/filters/RandomPathFilter.html"
  [& {:keys [rng extensions max-paths]
      :or {max-paths 0}}]
  (RandomPathFilter. rng (array-of :data extensions
                                   :java-type java.lang.String) max-paths))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input split user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-filesplit
  "File input split. Splits up a file into writables

  :path (str), the path to the file you want to import

  :rng-seed (java.util.Random), seed for consistent randomization

  :allow-format (collection of string(s)), the file formats allowed

  :recursive? (boolean), how the files should be read in

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/FileSplit.html"
  [& {:keys [path rng-seed allow-format recursive?]
      :as opts}]
  (input-split {:file-split opts}))

(defn new-collection-input-split
  "creates a new collection input split
   - A simple InputSplit based on a collection of URIs

  coll (coll of URIs), the URIs to import

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/CollectionInputSplit.html"
  [& {:keys [coll]}]
  (input-split {:collection-input-split {:collection coll}}))

(defn new-input-stream-input-split
  "Input stream input split.
  The normal pattern is reading the whole input stream and turning that in to a record.
  This is meant for streaming raw data rather than normal mini batch pre processing.

  :file-path can be a string, or a uri

  :in-stream, is the input stream

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/InputStreamInputSplit.html"
  [& {:keys [in-stream file-path]
      :as opts}]
  (input-split {:input-stream-input-split opts}))

(defn new-list-string-split
  "An input split that already has delimited data of some kind.

  :data should be a lists of strings

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/ListStringSplit.html"
  [& {:keys [data]
      :as opts}]
  (input-split {:list-string-split opts}))

(defn new-numbered-file-input-split
  "InputSplit for sequences of numbered files.

  ex. Suppose files are sequenced according to my_file_100.txt, my_file_101.txt,
      ..., my_file_200.txt  then use new-numbered-file-input-split as such:
      (new-numbered-file-input-split
       {:base-string my_file_%d.txt :inclusive-min-idx 100 :inclusive-max-idx 200})

  :base-string (str) regex for file names

  :inclusive-min-idx (int) starting index of files

  :inclusive-max-idx (int) end index of ffiles

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/NumberedFileInputSplit.html"
  [& {:keys [base-string inclusive-min-idx inclusive-max-idx]
      :as opts}]
  (input-split {:numbered-file-input-split opts}))

(defn new-string-split
  "String split used for single line inputs

   :data (str), a string to be used as data

   see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/StringSplit.html"
  [& {:keys [data]
      :as opts}]
  (input-split {:string-split opts}))

(defn new-transform-split
  "input-split implementation that maps the URIs of a given base-input-split to new URIs.
   - Useful when features and labels are in different files sharing a common naming scheme,
     and the name of the output file can be determined given the name of the input file.

  :base-input-split (input-split), an input split containing the URIs

  :to-be-repalced (str), a string to search through the uris in the input split

  :replaced-with (str), the string to replace what matches to-be-replaced

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/TransformSplit.html"
  [& {:keys [base-input-split to-be-replaced
             replaced-with]
      :as opts}]
  (input-split {:transform-split opts}))
