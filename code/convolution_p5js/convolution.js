// convolution.js
// Compatível com p5.js v2.0.5

let imgOriginal = null;
let imgFiltered = null;
let canvas;
let canvasWidth = 700;
let canvasHeight = 370;

const kernels = {
  "Identity": {
    size: 3,
    matrix: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]
    ]
  },
  "Blur 3x3": {
    size: 3,
    matrix: [
      [1,1,1],
      [1,1,1],
      [1,1,1]
    ]
  },
  "Gaussian 5x5": {
    size: 5,
    matrix: [
      [1,4,7,4,1],
      [4,16,26,16,4],
      [7,26,41,26,7],
      [4,16,26,16,4],
      [1,4,7,4,1]
    ]
  },
  "Sharpen": {
    size:3,
    matrix: [
      [0,-1,0],
      [-1,5,-1],
      [0,-1,0]
    ]
  },
  "Sobel X": {
    size:3,
    matrix: [
      [-1,0,1],
      [-2,0,2],
      [-1,0,1]
    ]
  },
  "Sobel Y": {
    size:3,
    matrix: [
      [-1,-2,-1],
      [0,0,0],
      [1,2,1]
    ]
  },
  "Laplacian": {
    size:3,
    matrix: [
      [0,-1,0],
      [-1,4,-1],
      [0,-1,0]
    ]
  }
};

let selectedKernelName = "Identity";
let kernelSize = 3;
let kernelMatrix = [];
let normalizeKernel = true;

let kernelInputs = [];

let animating = false;
let stepX = 0;
let stepY = 0;

function preload() {
  // Não carregar imagem fixa — imagem será carregada via input file
}

function setup() {
  // Criar canvas inicial com tamanho base
  canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.body);

  // Inicializar kernel padrão
  setKernel(kernels[selectedKernelName]);

  // Ligar elementos do DOM do HTML
  const selectKernel = document.getElementById('selectKernel');
  const chkNormalize = document.getElementById('chkNormalize');
  const btnApplyFull = document.getElementById('btnApplyFull');
  const btnStep = document.getElementById('btnStep');
  const btnReset = document.getElementById('btnReset');
  const kernelEditorDiv = document.getElementById('kernelEditor');
  
  // Popular select de kernels
  for (const k in kernels) {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = k;
    selectKernel.appendChild(opt);
  }
  
  selectKernel.value = selectedKernelName;

  selectKernel.addEventListener('change', () => {
    selectedKernelName = selectKernel.value;
    setKernel(kernels[selectedKernelName]);
    buildKernelEditor();
    imgFiltered = null;
    animating = false;
    redraw();
  });

  chkNormalize.checked = normalizeKernel;

  chkNormalize.addEventListener('change', () => {
    normalizeKernel = chkNormalize.checked;
    buildKernelEditor();
    imgFiltered = null;
    animating = false;
    redraw();
  });

  btnApplyFull.addEventListener('click', () => {
    if (!imgOriginal || animating) return;
    applyFilterFull();
  });

  btnReset.addEventListener('click', () => {
    if (!imgOriginal || animating) return;
    imgFiltered = null;
    animating = false;
    redraw();
  });

  btnStep.addEventListener('click', () => {
    if (!imgOriginal || animating) return startAnimation();
  });

  // Construir editor inicial
  buildKernelEditor();

  
  // Input para carregar imagem local
  const fileInput = document.getElementById('fileInput');
  
  fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    
    if (!file) return;

    const url = URL.createObjectURL(file);

    loadImage(url,
      img => {
        imgOriginal = img;
        imgFiltered = null;
        animating = false;

        // Ajustar tamanho do canvas para mostrar imagem + resultado lado a lado + margem
        let wNew = imgOriginal.width *2 +20;
        let hNew = imgOriginal.height +50;

        resizeCanvas(wNew,hNew);

        redraw();

        URL.revokeObjectURL(url);
      },
      err => {
        alert('Erro ao carregar a imagem.');
        console.error(err);
      }
    );
    
  });

  
  noLoop(); // só desenha quando necessário
}

function setKernel(kernelObj) {
  kernelMatrix = [];
  kernelSize = kernelObj.size;

  if (kernelSize !==3){
    kernelMatrix = kernelObj.matrix.map(row => row.slice());
    return;
  }

  for(let i=0; i<3; i++){
    kernelMatrix[i] = [];
    for(let j=0; j<3; j++){
      kernelMatrix[i][j] = kernelObj.matrix[i][j];
    }
  }
}

function buildKernelEditor() {
  
   const container = document.getElementById('kernelEditor');
   container.innerHTML='';
   
   const title = document.createElement('p');
   title.textContent='Editor do Kernel 3x3:';
   container.appendChild(title);
   
   if(kernelSize !==3){
     const msg=document.createElement('div');
     msg.textContent='Editor disponível apenas para kernels de dimensão 3x3.';
     container.appendChild(msg);
     kernelInputs=[];
     return;
   }
   
   const table=document.createElement('div');
   table.style.display='inline-block';
   table.style.marginTop='6px';
   
   kernelInputs=[];
   
   for(let i=0;i<3;i++){
     const rowDiv=document.createElement('div');
     rowDiv.style.marginBottom='4px';
     kernelInputs[i]=[];
     for(let j=0;j<3;j++){
       const inp=document.createElement('input');
       inp.type='number';
       inp.step='0.1';
       inp.value=kernelMatrix[i][j]!==undefined ? kernelMatrix[i][j] : '0';
       inp.style.width='50px';
       inp.style.marginRight='6px';
       inp.addEventListener('input',()=>{
         let v=parseFloat(inp.value);
         if(Number.isNaN(v)) v=0;
         kernelMatrix[i][j]=v;
         imgFiltered=null;
         animating=false;
         redraw();
       });
       rowDiv.appendChild(inp);
       kernelInputs[i][j]=inp;
     }
     table.appendChild(rowDiv);
   }
   
   container.appendChild(table);
   
   // Soma do kernel e indicação de normalização
   const sumDiv=document.createElement('div');
   sumDiv.style.marginTop='6px';
   const sumVal=kernelMatrix.flat().reduce((a,b)=>a+(+b),0);
   sumDiv.textContent=`Soma do kernel: ${sumVal.toFixed(3)} (${normalizeKernel ? 'a ser normalizado' : 'não normalizado'})`;
   
   container.appendChild(sumDiv);
}

function normalizeKernelMatrix(matrix) {
   const sum=matrix.flat().reduce((a,b)=>a+(+b),0);
   if(sum===0) return matrix.map(row=>row.slice());
   return matrix.map(row=>row.map(v=>v/sum));
}

function applyConvolution(img,kernel,norm=true,xStep=null,yStep=null){
   const w=img.width;
   const h=img.height;

   const result=createImage(w,h);
   result.loadPixels();
   img.loadPixels();

   const kSize=kernel.length;
   const kHalf=Math.floor(kSize/2);

   const k=norm ? normalizeKernelMatrix(kernel):kernel;

   if(xStep!==null && yStep!==null){
     for(let i=0;i<img.pixels.length;i++) result.pixels[i]=img.pixels[i];

     let rSum=0,gSum=0,bSum=0;

     for(let ky=0;ky<kSize;ky++){
       for(let kx=0;kx<kSize;kx++){
         let px=xStep+kx-kHalf;
         let py=yStep+ky-kHalf;

         px=constrain(px,0,w-1);
         py=constrain(py,0,h-1);

         const idx=(py*w+px)*4;

         const weight=k[ky][kx];

         rSum+=img.pixels[idx]*weight;
         gSum+=img.pixels[idx+1]*weight;
         bSum+=img.pixels[idx+2]*weight;
       }
     }

     const idxOut=(yStep*w+xStep)*4;

     result.pixels[idxOut]=constrain(rSum,0,255);
     result.pixels[idxOut+1]=constrain(gSum,0,255);
     result.pixels[idxOut+2]=constrain(bSum,0,255);
     result.pixels[idxOut+3]=255;

     result.updatePixels();

     return result;
   }

   for(let y=0;y<h;y++){
     for(let x=0;x<w;x++){
       let rSum=0,gSum=0,bSum=0;

       for(let ky=0;ky<kSize;ky++){
         for(let kx=0;kx<kSize;kx++){

           let px=x+kx-kHalf;
           let py=y+ky-kHalf;

           px=constrain(px,0,w-1);
           py=constrain(py,0,h-1);

           const idx=(py*w+px)*4;

           const weight=k[ky][kx];

           rSum+=img.pixels[idx]*weight;
           gSum+=img.pixels[idx+1]*weight;
           bSum+=img.pixels[idx+2]*weight;

         }
       }

       const idxOut=(y*w+x)*4;

       result.pixels[idxOut]=constrain(rSum,0,255);
       result.pixels[idxOut+1]=constrain(gSum,0,255);
       result.pixels[idxOut+2]=constrain(bSum,0,255);
       result.pixels[idxOut+3]=255;

     }
   }

   result.updatePixels();
   return result;
}

function applyFilterFull(){
   if(!imgOriginal) return;

   if(kernelSize !==3){
     imgFiltered=applyConvolution(imgOriginal,kernelMatrix,normalizeKernel);
     animating=false;
     redraw();
     return;
   }

   imgFiltered=applyConvolution(imgOriginal,kernelMatrix,normalizeKernel);
   animating=false;
   redraw();
}

function startAnimation(){
   if(!imgOriginal) return;

   if(kernelSize !==3){
     alert("A animação passo a passo está disponível apenas para kernels de dimensão 3x3.");
     return;
   }

   animating=true;

   imgFiltered=createImage(imgOriginal.width,imgOriginal.height);
   imgFiltered.copy(imgOriginal,0,0,imgOriginal.width,imgOriginal.height,
                    0,0,imgOriginal.width,imgOriginal.height);

   stepX=0;
   stepY=0;

   loop();
}

function draw(){
   background(220);

   if(imgOriginal){
     image(imgOriginal,0,0,imgOriginal.width,imgOriginal.height);
     fill(0);
     noStroke();
     textSize(12);
     text("Imagem Original",10,imgOriginal.height+14);
   }

   if(imgFiltered){
     image(imgFiltered,imgOriginal.width+10,0,imgFiltered.width,imgFiltered.height);
     fill(0);
     noStroke();
     textSize(12);
     text("Imagem Filtrada",imgOriginal.width+10,imgFiltered.height+14);
   }

   if(animating && kernelSize ===3){
     drawKernelWindow(stepX,stepY);

     showConvolutionCalculation(stepX,stepY);

     imgFiltered=applyConvolution(imgOriginal,kernelMatrix,
                                 normalizeKernel,
                                 stepX,
                                 stepY);

     advanceStep();

   } else {
     const calcDiv=document.getElementById('calculationText');
     if(calcDiv) calcDiv.style.display='none';
     noLoop();
   }
}

function drawKernelWindow(x,y){
   push();
   noFill();
   stroke(255,0,0);
   strokeWeight(2);

   rect(x - Math.floor(kernelSize/2) + .5,
        y - Math.floor(kernelSize/2) + .5,
        kernelSize,
        kernelSize);

   pop();
}

function showConvolutionCalculation(x,y){
   const calcDiv=document.getElementById('calculationText');
   if(!calcDiv) return;

   const kHalf=Math.floor(kernelSize/2);
   
   const normK=normalizeKernel ? normalizeKernelMatrix(kernelMatrix) : kernelMatrix;

   let lines=[];
   
   let rSum=0,gSum=0,bSum=0;

   lines.push(`Cálculo convolução para pixel (${x},${y}):`);
   lines.push('-----------------------------------------');

   for(let ky=0;ky<kernelSize;ky++){
     for(let kx=0;kx<kernelSize;kx++){
       let px=x+kx-kHalf;
       let py=y+ky-kHalf;

       px=constrain(px,0,imgOriginal.width-1);
       py=constrain(py,0,imgOriginal.height-1);

       const idx=(py*imgOriginal.width + px)*4;

       const weight=normK[ky][kx];

       const rVal=imgOriginal.pixels[idx];
       const gVal=imgOriginal.pixels[idx+1];
       const bVal=imgOriginal.pixels[idx+2];

       rSum+=rVal*weight;
       gSum+=gVal*weight;
       bSum+=bVal*weight;

       lines.push(`Pos(${px},${py}) RGB(${rVal},${gVal},${bVal}) * ${weight.toFixed(3)}`);
     }
   }

   lines.push('-----------------------------------------');
   lines.push(`Soma R=${rSum.toFixed(2)}, G=${gSum.toFixed(2)}, B=${bSum.toFixed(2)}`);

   calcDiv.style.display='block';
   calcDiv.textContent=lines.join('\n');
}

function advanceStep(){
   stepX++;
   
   if(stepX >= imgOriginal.width){
     stepX=0;
     stepY++;
     
     if(stepY >= imgOriginal.height){
       animating=false;

       imgFiltered=applyConvolution(imgOriginal,kernelMatrix,
                                   normalizeKernel);

       noLoop();
     }
   }
}
