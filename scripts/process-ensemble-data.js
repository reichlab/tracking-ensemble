// Script to convert ensemble files from cdc-flusight-ensemble to numpy
// matrices for ingestion

const fct = require('flusight-csv-tools')
const fs = require('fs-extra')
const path = require('path')
const u = require('./utils')

if (process.argv.length < 4) {
  console.log(`Usage: node <script-name> <input-dir> <output-dir>`)
  process.exit(1)
}

const INPUT_DIR = process.argv[2]
const OUTPUT_DIR = process.argv[3]

async function dumpData (modelData, filterIndices) {
  await fs.ensureDir(OUTPUT_DIR)

  // Write the common index using data from any model
  let commonIndices = filterIndices[0].map(fi => modelData[0].indices[fi])
  let indexLines = ['epiweek,region', ...(commonIndices.map(it => it.join(',')))]
  u.writeLines(indexLines, path.join(OUTPUT_DIR, 'index.csv'))

  for (let mIdx = 0; mIdx < modelData.length; mIdx++) {
    // For each model data
    let outputModelDir = path.join(OUTPUT_DIR, modelData[mIdx].name)
    await fs.ensureDir(outputModelDir)

    for (let target of fct.meta.targetIds) {
      let targetPath = path.join(outputModelDir, target)
      let filteredArray = filterIndices[mIdx].map(fi => modelData[mIdx].predictions[target][fi])
      u.npSaveTxt(filteredArray, targetPath)
    }
  }
}

function processModels (models) {
  let data = []

  models.forEach((model, mIdx) => {
    let csvs = model.csvs
    data[mIdx] = { name: model.id, indices: [], predictions: {} }

    csvs.forEach((csv) => {
      try {
        let epiweek = u.getCsvEpiweek(csv)
        let fctCsv = new fct.Csv(csv, epiweek, model.id)
        for (let target of fct.meta.targetIds) {
          if (data[mIdx].predictions[target] === undefined) {
            data[mIdx].predictions[target] = []
          }
          for (let region of fct.meta.regionIds) {
            let probs = fctCsv.getBins(target, region).map(b => b[2])
            // Take 34 bins for peak-wk and 35 for onset-wk irrespective of season
            if ((target === 'peak-wk') && (probs.length === 33)) {
              probs.push(0.0)
            } else if ((target === 'onset-wk') && (probs.length === 34)) {
              probs.splice(probs.length - 1, 0, 0.0)
            }
            data[mIdx].predictions[target].push(probs)
            data[mIdx].indices.push([epiweek, region])
          }
        }
      } catch (e) {
        console.log(`Error in parsing ${csv}`)
      }
    })
    console.log(`Model ${model.id} parsed`)
  })

  return data
}

/**
 * Filter common items from the epiweek-region indices
 */
function getFilterIndices (indices) {
  let comparator = (a, b) => {
    // Revert the items to original file name ordering
    let indToF = i => `${(i[0] + '').slice(4)}-${Math.floor(i[0] / 100)}-${i[1]}`
    let fa = indToF(a)
    let fb = indToF(b)
    if (fa === fb) {
      return 0
    } else if (fa > fb) {
      return 1
    } else {
      return -1
    }
  }

  return u.intersectionIndices(indices, comparator)
}

let modelData = processModels(u.getModels(INPUT_DIR))
let filterIndices = getFilterIndices(modelData.map(md => md.indices))
dumpData(modelData, filterIndices).then(() => console.log('All done'))
