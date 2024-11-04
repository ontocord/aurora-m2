import time
from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM
#from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
from torch import multiprocessing

import logging
logger = logging.getLogger(__name__)  

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.WARNING)


model_name = "Lyte/Llama-3.2-3B-Overthinker"
model_name = "NousResearch/Hermes-3-Llama-3.1-8B"
#"flash_attention_2"
#"sdpa"



from torchao.prototype.low_bit_optim import CPUOffloadOptimizer, AdamW8bit
import torchao

text = """Abraham Lincoln (/ˈlɪŋkən/ LINK-ən; February 12, 1809 – April 15, 1865) was the 16th president of the United States, serving from 1861 until his assassination in 1865. He led the United States through the American Civil War, defending the nation as a constitutional union, defeating the Confederacy, playing a major role in the abolition of slavery, expanding the power of the federal government, and modernizing the U.S. economy.

Lincoln was born into poverty in a log cabin in Kentucky and was raised on the frontier, mainly in Indiana. He was self-educated and became a lawyer, Whig Party leader, Illinois state legislator, and U.S. representative from Illinois. In 1849, he returned to his successful law practice in Springfield, Illinois. In 1854, angered by the Kansas–Nebraska Act, which opened the territories to slavery, he re-entered politics. He soon became a leader of the new Republican Party. He reached a national audience in the 1858 Senate campaign debates against Stephen A. Douglas. Lincoln ran for president in 1860, sweeping the North to gain victory. Pro-slavery elements in the South viewed his election as a threat to slavery, and Southern states began seceding from the nation. They formed the Confederate States of America, which began seizing federal military bases in the South. A little over one month after Lincoln assumed the presidency, Confederate forces attacked Fort Sumter, a U.S. fort in South Carolina. Following the bombardment, Lincoln mobilized forces to suppress the rebellion and restore the union.

Lincoln, a moderate Republican, had to navigate a contentious array of factions with friends and opponents from both the Democratic and Republican parties. His allies, the War Democrats and the Radical Republicans, demanded harsh treatment of the Southern Confederates. He managed the factions by exploiting their mutual enmity, carefully distributing political patronage, and by appealing to the American people. Anti-war Democrats (called "Copperheads") despised Lincoln, and some irreconcilable pro-Confederate elements went so far as to plot his assassination. His Gettysburg Address became one of the most famous speeches in American history. Lincoln closely supervised the strategy and tactics in the war effort, including the selection of generals, and implemented a naval blockade of the South's trade. He suspended habeas corpus in Maryland and elsewhere, and he averted war with Britain by defusing the Trent Affair. In 1863, he issued the Emancipation Proclamation, which declared the slaves in the states "in rebellion" to be free. It also directed the Army and Navy to "recognize and maintain the freedom of said persons" and to receive them "into the armed service of the United States." Lincoln pressured border states to outlaw slavery, and he promoted the Thirteenth Amendment to the U.S. Constitution, which abolished slavery, except as punishment for a crime. Lincoln managed his own successful re-election campaign. He sought to heal the war-torn nation through reconciliation. On April 14, 1865, just five days after the Confederate surrender at Appomattox, he was attending a play at Ford's Theatre in Washington, D.C., with his wife, Mary, when he was fatally shot by Confederate sympathizer John Wilkes Booth.

Lincoln is remembered as a martyr and a national hero for his wartime leadership and for his efforts to preserve the Union and abolish slavery. Lincoln is often ranked in both popular and scholarly polls as the greatest president in American history.

Abraham Lincoln was born on February 12, 1809, the second child of Thomas Lincoln and Nancy Hanks Lincoln, in a log cabin on Sinking Spring Farm near Hodgenville, Kentucky.[2] He was a descendant of Samuel Lincoln, an Englishman who migrated from Hingham, Norfolk, to its namesake, Hingham, Massachusetts, in 1638. The family then migrated west, passing through New Jersey, Pennsylvania, and Virginia.[3] Lincoln was also a descendant of the Harrison family of Virginia; his paternal grandfather and namesake, Captain Abraham Lincoln and wife Bathsheba (née Herring) moved the family from Virginia to Jefferson County, Kentucky.[b] The captain was killed in an Indian raid in 1786.[5] His children, including eight-year-old Thomas, Abraham's father, witnessed the attack.[6][c] Thomas then worked at odd jobs in Kentucky and Tennessee before the family settled in Hardin County, Kentucky, in the early 1800s.[6]

The farm site where Lincoln grew up in Spencer County, Indiana
Lincoln's mother Nancy Lincoln is widely assumed to be the daughter of Lucy Hanks.[8] Through his mother’s family, he is related to American actor Tom Hanks. Thomas and Nancy married on June 12, 1806, in Washington County, and moved to Elizabethtown, Kentucky.[9] They had three children: Sarah, Abraham, and Thomas, who died as an infant.[10]

Thomas Lincoln bought multiple farms in Kentucky, but could not get clear titles to any, losing hundreds of acres of land in property disputes.[11] In 1816, the family moved to Indiana, where the land surveys and titles were more reliable.[12] They settled in an "unbroken forest"[13] in Hurricane Township, Perry County, Indiana.[14] When the Lincolns moved to Indiana it "had just been admitted to the Union" as a "free" (non-slaveholding) state,[15] except that, though "no new enslaved people were allowed, ... currently enslaved individuals remained so".[16][d] In 1860, Lincoln noted that the family's move to Indiana was "partly on account of slavery", but mainly due to land title difficulties.[18][19]

In Kentucky and Indiana, Thomas worked as a farmer, cabinetmaker, and carpenter.[20] At various times he owned farms, livestock, and town lots, paid taxes, sat on juries, appraised estates, and served on county patrols. Thomas and Nancy were members of the Separate Baptist Church, "condemned profanity, intoxication, gossip, horse racing, and dancing." Most of its members opposed slavery.[21]

Overcoming financial challenges, Thomas in 1827 obtained clear title to 80 acres (32 ha) in Indiana, an area that became known as Little Pigeon Creek Community.[22]

Mother's death
On October 5, 1818, Nancy Lincoln died from milk sickness, leaving 11-year-old Sarah in charge of a household including her father, nine-year-old Abraham, and Nancy's 19-year-old orphan cousin, Dennis Hanks.[23] Ten years later, on January 20, 1828, Sarah died while giving birth to a stillborn son, devastating Lincoln.[24]

On December 2, 1819, Thomas married Sarah Bush Johnston, a widow from Elizabethtown, Kentucky, with three children of her own.[25] Abraham became close to his stepmother and called her "Mother".[26] Dennis Hanks said he was lazy, for all his "reading—scribbling—writing—ciphering—writing poetry".[27] His stepmother acknowledged he did not enjoy "physical labor" but loved to read.[28][29]

Education and move to Illinois
Lincoln was largely self-educated.[30] His formal schooling was from itinerant teachers. It included two short stints in Kentucky, where he learned to read, but probably not to write. In Indiana at age seven,[31] due to farm chores, he attended school only sporadically, for a total of fewer than 12 months in aggregate by age 15.[32] Nonetheless, he remained an avid reader and retained a lifelong interest in learning.[33] Family, neighbors, and schoolmates recalled that his readings included the King James Bible, Aesop's Fables, John Bunyan's The Pilgrim's Progress, Daniel Defoe's Robinson Crusoe, and The Autobiography of Benjamin Franklin.[34] Despite being self-educated, Lincoln was the recipient of honorary degrees later in life, including an honorary Doctor of Laws from Columbia University in June 1861.[35]

When Lincoln was a teen, his "father grew more and more to depend on him for the 'farming, grubbing, hoeing, making fences' necessary to keep the family afloat. He also regularly hired his son out to work ... and by law, he was entitled to everything the boy earned until he came of age".[36] Lincoln was tall, strong, and athletic, and became adept at using an ax.[37] He was an active wrestler during his youth and trained in the rough catch-as-catch-can style (also known as catch wrestling). He became county wrestling champion at the age of 21.[38] He gained a reputation for his strength and audacity after winning a wrestling match with the renowned leader of ruffians known as the Clary's Grove boys.[39]

In March 1830, fearing another milk sickness outbreak, several members of the extended Lincoln family, including Abraham, moved west to Illinois, a free state, and settled in Macon County.[40][e] Abraham then became increasingly distant from Thomas, in part, due to his father's lack of interest in education.[42] In 1831, as Thomas and other family members prepared to move to a new homestead in Coles County, Illinois, Abraham struck out on his own.[43] He made his home in New Salem, Illinois, for six years.[44] Lincoln and some friends took goods, including live hogs, by flatboat to New Orleans, Louisiana, where he first witnessed slavery.[45]
"""

text = """Abraham Lincoln (/ˈlɪŋkən/ LINK-ən; February 12, 1809 – April 15, 1865) was the 16th president of the United States, serving from 1861 until his assassination in 1865. He led the United States through the American Civil War, defending the nation as a constitutional union, defeating the Confederacy, playing a major role in the abolition of slavery, expanding the power of the federal government, and modernizing the U.S. economy.

Lincoln was born into poverty in a log cabin in Kentucky and was raised on the frontier, mainly in Indiana. He was self-educated and became a lawyer, Whig Party leader, Illinois state legislator, and U.S. representative from Illinois. In 1849, he returned to his successful law practice in Springfield, Illinois. In 1854, angered by the Kansas–Nebraska Act, which opened the territories to slavery, he re-entered politics. He soon became a leader of the new Republican Party. He reached a national audience in the 1858 Senate campaign debates against Stephen A. Douglas. Lincoln ran for president in 1860, sweeping the North to gain victory. Pro-slavery elements in the South viewed his election as a threat to slavery, and Southern states began seceding from the nation. They formed the Confederate States of America, which began seizing federal military bases in the South. A little over one month after Lincoln assumed the presidency, Confederate forces attacked Fort Sumter, a U.S. fort in South Carolina. Following the bombardment, Lincoln mobilized forces to suppress the rebellion and restore the union.

Lincoln, a moderate Republican, had to navigate a contentious array of factions with friends and opponents from both the Democratic and Republican parties. His allies, the War Democrats and the Radical Republicans, demanded harsh treatment of the Southern Confederates. He managed the factions by exploiting their mutual enmity, carefully distributing political patronage, and by appealing to the American people. Anti-war Democrats (called "Copperheads") despised Lincoln, and some irreconcilable pro-Confederate elements went so far as to plot his assassination. His Gettysburg Address became one of the most famous speeches in American history. Lincoln closely supervised the strategy and tactics in the war effort, including the selection of generals, and implemented a naval blockade of the South's trade. He suspended habeas corpus in Maryland and elsewhere, and he averted war with Britain by defusing the Trent Affair. In 1863, he issued the Emancipation Proclamation, which declared the slaves in the states "in rebellion" to be free. It also directed the Army and Navy to "recognize and maintain the freedom of said persons" and to receive them "into the armed service of the United States." Lincoln pressured border states to outlaw slavery, and he promoted the Thirteenth Amendment to the U.S. Constitution, which abolished slavery, except as punishment for a crime. Lincoln managed his own successful re-election campaign. He sought to heal the war-torn nation through reconciliation. On April 14, 1865, just five days after the Confederate surrender at Appomattox, he was attending a play at Ford's Theatre in Washington, D.C., with his wife, Mary, when he was fatally shot by Confederate sympathizer John Wilkes Booth.

Lincoln is remembered as a martyr and a national hero for his wartime leadership and for his efforts to preserve the Union and abolish slavery. Lincoln is often ranked in both popular and scholarly polls as the greatest president in American history.

Abraham Lincoln was born on February 12, 1809, the second child of Thomas Lincoln and Nancy Hanks Lincoln, in a log cabin on Sinking Spring Farm near Hodgenville, Kentucky.[2] He was a descendant of Samuel Lincoln, an Englishman who migrated from Hingham, Norfolk, to its namesake, Hingham, Massachusetts, in 1638. The family then migrated west, passing through New Jersey, Pennsylvania, and Virginia.[3] Lincoln was also a descendant of the Harrison family of Virginia; his paternal grandfather and namesake, Captain Abraham Lincoln and wife Bathsheba (née Herring) moved the family from Virginia to Jefferson County, Kentucky.[b] The captain was killed in an Indian raid in 1786.[5] His children, including eight-year-old Thomas, Abraham's father, witnessed the attack.[6][c] Thomas then worked at odd jobs in Kentucky and Tennessee before the family settled in Hardin County, Kentucky, in the early 1800s.[6]

"""
cache_dir = "/leonardo_work/EUHPC_E03_068/.cache/"
#3b can do at around 13K sequence
#7b can do at 5K sequence
optim = tokenizer = model = None
device_no = -1
def process_one(device, seq_len_mult=8, batch_size=1, grad_accum_step=1, num_iter=100):
    global tokenizer, model, device_no
    logger.warning(device)
    if device_no == -1:
        device_no = device
    try:
        if model is None: assert False
    except:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, cache_dir=cache_dir).to('cuda:'+str(device_no)).train()
        tokenizer = AutoTokenizer.from_pretrained(model_name, left_padding=True, cache_dir=cache_dir)
        optim = None
        logger.warning("loaded tokenizer")
        optim = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW,   offload_gradients=True, fused=True) # torch.optim.AdamW, fused=True,
        logger.warning("loaded optim")
        #optim = CPUOffloadOptimizer(model.parameters(), AdamW8bit, offload_gradients=True, ) # , fused=True
    logger.warning(model.device)
    final_loss = None    
    text2 = text* seq_len_mult
    loss = None
    t0 = time.time()
    num_tokens = len(tokenizer(text2)['input_ids'])
    for i in range(num_iter):
      batch = tokenizer([text2]*batch_size, return_tensors="pt", truncation=True, padding=True).to('cuda:'+str(device_no))
      # set certain labels to -100
      batch['labels'] = batch['input_ids']
      output = model(**batch)
      if loss is None:
        loss = output.loss
      else:
        loss += output.loss
      optim.step()
      batch = None
      output = None
      if (i+1)%grad_accum_step == 0:
        loss.backward() # retain_graph=False)
        optim.zero_grad()
        optim.step()
        loss = final_loss = float(loss.item())        
        logger.warning (f"step {i} " + str( num_tokens*batch_size*(i+1)/(time.time()-t0)) + " " + str(loss) + " " + str(num_tokens))
        loss = None
    return final_loss

num_devices = torch.cuda.device_count()        
if __name__ == "__main__":

    multiprocessing.set_start_method('spawn', force=True)
    with  multiprocessing.Pool(processes=num_devices) as pool:
        for loss in pool.map(process_one, list(range(num_devices))):
            logger.warning (loss)
