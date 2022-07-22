function ajaxpost () {
  // (A) GET FORM DATA
  var form = document.getElementById("myForm");
  var object = {};
  var link = "";
  var iframe = document.createElement('iframe');
  var text = "";
  iframe.frameBorder=0;
  iframe.width="833px";
  iframe.height="901px";
  iframe.id="randomid";
  var formData = new FormData(document.querySelector('form'));
  var formDataObject = Object.fromEntries(formData.entries());
  // Format the plain form data as JSON
  var formDataJsonString = JSON.stringify(formDataObject);
  console.log(formDataJsonString)
  // var formData = new FormData(document.querySelector('form'))
  // formData.forEach((value, key) => object[key] = value);
  // var json = JSON.stringify(object);
  // console.log(json)
 
  // (B) AJAX REQUEST
  // (B1) POST DATA TO SERVER, RETURN RESPONSE AS TEXT
  fetch("https://vyfeovv08c.execute-api.us-east-2.amazonaws.com/beta", { method:"POST", headers: {
    'Content-Type': 'application/json;charset=utf-8'
  }, body:formDataJsonString})
  .then(response => response.json())
  .then(data => {
    console.log(data.body)
    if (data.body.Prediction == "0") { link =  "https://c.tenor.com/3rLBWyUwf10AAAAC/artificilbrain-ai.gif"; text = "Based on a predictive model trained on stroke patient data from around 5,000 patients, your risk of stroke is LOW."; }
    else { link =  "https://c.tenor.com/Xvxg2q3dJ5gAAAAC/ai-brain.gif"; text = "Based on a predictive model trained on stroke patient data from around 5,000 patients, your risk of stroke is HIGH. Preventative measures against a stroke can be found here: https://www.cdc.gov/stroke/prevention.htm" ;}
  iframe.setAttribute("src", link);
  predictionDiv = document.getElementById("strokeRiskResult") ;
  predictionDiv.innerHTML = '';
  predictionDiv.appendChild(iframe);
  predictionDiv.insertAdjacentText('beforebegin', text);
  })
  // (B3) OPTIONAL - HANDLE FETCH ERROR
  .catch((err) => { console.error(err); });
 
  // (C) PREVENT FORM SUBMIT
  return false;
}
